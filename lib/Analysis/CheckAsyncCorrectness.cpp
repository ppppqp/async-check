//===- CheckAsyncCorrectness.cpp - GPU correctness checker ------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsyncCheck/Analysis/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace async_check {
namespace {

using namespace mlir;

enum class Overlap { No, Yes, Unknown };
enum class Uniformity { Uniform, Divergent, Unknown };

struct AccessRegion {
  Value base;
  SmallVector<Value> indices;
  int64_t elements = 1;
  Operation *origin = nullptr;
};

struct PendingCopy {
  AccessRegion destination;
  nvgpu::DeviceAsyncCopyOp copy;
  bool complete = false;
};

struct CopyGroup {
  SmallVector<unsigned> copies;
};

static std::optional<int64_t> getConstantInteger(Value value) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant)))
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant);
  if (!integer)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

static bool hasWorkgroupMemorySpace(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(type);
}

static bool haveIdenticalIndices(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Compare direct regions. The first implementation deliberately avoids
/// pretending that it has general alias or Presburger reasoning.
static Overlap classifyOverlap(const AccessRegion &lhs,
                               const AccessRegion &rhs) {
  if (lhs.base != rhs.base)
    return Overlap::No;
  if (lhs.elements <= 0 || rhs.elements <= 0)
    return Overlap::No;
  if (haveIdenticalIndices(lhs.indices, rhs.indices))
    return Overlap::Yes;

  auto type = dyn_cast<MemRefType>(lhs.base.getType());
  if (!type || type.getRank() != 1 || lhs.indices.size() != 1 ||
      rhs.indices.size() != 1)
    return Overlap::Unknown;

  std::optional<int64_t> lhsStart = getConstantInteger(lhs.indices.front());
  std::optional<int64_t> rhsStart = getConstantInteger(rhs.indices.front());
  if (!lhsStart || !rhsStart)
    return Overlap::Unknown;

  int64_t lhsEnd = *lhsStart + lhs.elements;
  int64_t rhsEnd = *rhsStart + rhs.elements;
  return *lhsStart < rhsEnd && *rhsStart < lhsEnd ? Overlap::Yes
                                                  : Overlap::No;
}

static bool isConstantLike(Value value) {
  return getConstantInteger(value).has_value();
}

/// Recognize the small injective ownership subset used to suppress warnings
/// for store shared[tid] followed by load shared[tid].
static bool isUniqueThreadIndex(Value value) {
  if (isa_and_nonnull<gpu::ThreadIdOp>(value.getDefiningOp()))
    return true;

  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return isUniqueThreadIndex(cast.getIn());

  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return (isUniqueThreadIndex(add.getLhs()) && isConstantLike(add.getRhs())) ||
           (isUniqueThreadIndex(add.getRhs()) && isConstantLike(add.getLhs()));

  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return isUniqueThreadIndex(sub.getLhs()) && isConstantLike(sub.getRhs());

  return false;
}

static bool isSameThreadOwnedAccess(const AccessRegion &store,
                                    const AccessRegion &load) {
  return haveIdenticalIndices(store.indices, load.indices) &&
         llvm::any_of(store.indices, isUniqueThreadIndex);
}

static std::optional<unsigned> getThreadBound(gpu::ThreadIdOp threadId) {
  if (std::optional<llvm::APInt> upperBound = threadId.getUpperBound())
    return upperBound->getZExtValue();

  return gpu::getKnownDimensionSizeAround(
      threadId, gpu::DimensionKind::Block, threadId.getDimension());
}

static Uniformity classifyCmpUniformity(arith::CmpIOp compare,
                                        gpu::GPUFuncOp function) {
  Value varying = compare.getLhs();
  Value constantValue = compare.getRhs();
  bool varyingOnLeft = true;

  auto threadId = varying.getDefiningOp<gpu::ThreadIdOp>();
  if (!threadId) {
    varying = compare.getRhs();
    constantValue = compare.getLhs();
    varyingOnLeft = false;
    threadId = varying.getDefiningOp<gpu::ThreadIdOp>();
  }
  if (!threadId)
    return Uniformity::Unknown;

  std::optional<int64_t> constant = getConstantInteger(constantValue);
  std::optional<unsigned> bound = getThreadBound(threadId);
  if (!constant || !bound || *bound == 0)
    return Uniformity::Unknown;

  bool firstResult = false;
  for (unsigned id = 0; id < *bound; ++id) {
    unsigned bitWidth = 64;
    llvm::APInt idValue(bitWidth, id);
    llvm::APInt constantAP(bitWidth, static_cast<uint64_t>(*constant), true);
    bool result = varyingOnLeft
                      ? arith::applyCmpPredicate(compare.getPredicate(), idValue,
                                                 constantAP)
                      : arith::applyCmpPredicate(compare.getPredicate(),
                                                 constantAP, idValue);
    if (id == 0)
      firstResult = result;
    else if (result != firstResult)
      return Uniformity::Divergent;
  }
  return Uniformity::Uniform;
}

static Uniformity classifyUniformity(Value condition,
                                     gpu::GPUFuncOp function) {
  if (getConstantInteger(condition))
    return Uniformity::Uniform;
  if (auto compare = condition.getDefiningOp<arith::CmpIOp>())
    return classifyCmpUniformity(compare, function);
  return Uniformity::Unknown;
}

static bool hasProvenUniformParticipation(gpu::BarrierOp barrier) {
  if (barrier.getScope() != gpu::BarrierScope::Workgroup ||
      barrier.getNamedBarrier())
    return false;
  auto function = barrier->getParentOfType<gpu::GPUFuncOp>();
  if (!function)
    return false;
  for (Operation *parent = barrier->getParentOp();
       parent && parent != function.getOperation();
       parent = parent->getParentOp()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent))
      if (classifyUniformity(ifOp.getCondition(), function) !=
          Uniformity::Uniform)
        return false;
  }
  return true;
}

static bool checkDivergentBarriers(gpu::GPUFuncOp function) {
  bool foundError = false;
  function.walk([&](gpu::BarrierOp barrier) {
    if (barrier.getScope() != gpu::BarrierScope::Workgroup ||
        barrier.getNamedBarrier())
      return;

    scf::IfOp unknownIf;
    for (Operation *parent = barrier->getParentOp();
         parent && parent != function.getOperation();
         parent = parent->getParentOp()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (!ifOp)
        continue;

      Uniformity uniformity =
          classifyUniformity(ifOp.getCondition(), function);
      if (uniformity == Uniformity::Unknown) {
        if (!unknownIf)
          unknownIf = ifOp;
        continue;
      }
      if (uniformity != Uniformity::Divergent)
        continue;

      InFlightDiagnostic diagnostic = barrier.emitError(
          "workgroup barrier is control-divergent; not every thread reaches "
          "the same barrier instance");
      diagnostic.attachNote(ifOp.getCondition().getLoc())
          << "this control predicate varies across the known block dimensions";
      foundError = true;
      return;
    }
    if (unknownIf)
      barrier.emitRemark(
          "async-check: unknown whether every workgroup thread reaches this "
          "barrier");
  });
  return foundError;
}

class StraightLineBlockChecker {
public:
  bool check(Region &region) {
    for (Block &block : region) {
      checkBlock(block);
      for (Operation &operation : block)
        for (Region &nested : operation.getRegions())
          check(nested);
    }
    return foundError;
  }

private:
  void markGroupComplete(unsigned groupIndex) {
    for (unsigned copyIndex : groups[groupIndex].copies) {
      if (!copies[copyIndex].complete)
        storesSinceBarrier.push_back(copies[copyIndex].destination);
      copies[copyIndex].complete = true;
    }
  }

  void handleCopy(nvgpu::DeviceAsyncCopyOp copy) {
    AccessRegion destination{copy.getDst(),
                             SmallVector<Value>(copy.getDstIndices()),
                             copy.getDstElements().getSExtValue(), copy};

    for (const PendingCopy &pending : copies) {
      if (pending.complete)
        continue;
      Overlap overlap = classifyOverlap(destination, pending.destination);
      if (overlap == Overlap::Unknown) {
        copy.emitRemark("async-check: unknown whether this destination overlaps "
                        "an incomplete asynchronous copy");
      } else if (overlap == Overlap::Yes) {
        InFlightDiagnostic diagnostic = copy.emitError(
            "async-copy destination overlaps a previous incomplete "
            "asynchronous copy");
        diagnostic.attachNote(pending.copy->getLoc())
            << "previous asynchronous copy was issued here";
        foundError = true;
      }
    }

    unsigned index = copies.size();
    copies.push_back(PendingCopy{destination, copy, false});
    copyTokens[copy.getAsyncToken()] = index;
  }

  void handleGroup(nvgpu::DeviceAsyncCreateGroupOp group) {
    CopyGroup copyGroup;
    for (Value token : group.getInputTokens()) {
      auto it = copyTokens.find(token);
      if (it != copyTokens.end())
        copyGroup.copies.push_back(it->second);
    }
    unsigned index = groups.size();
    groups.push_back(std::move(copyGroup));
    groupTokens[group.getAsyncToken()] = index;
  }

  void handleWait(nvgpu::DeviceAsyncWaitOp wait) {
    auto groupIt = groupTokens.find(wait.getAsyncDependencies());
    if (!wait.getNumGroups()) {
      if (groupIt == groupTokens.end())
        return;
      // Groups execute in creation order. Completion of this group also proves
      // completion of every earlier group in this straight-line block.
      for (unsigned i = 0; i <= groupIt->second; ++i)
        markGroupComplete(i);
      return;
    }

    uint32_t groupsAllowedPending = *wait.getNumGroups();
    unsigned groupsKnownComplete =
        groups.size() > groupsAllowedPending
            ? groups.size() - static_cast<unsigned>(groupsAllowedPending)
            : 0;
    for (unsigned i = 0; i < groupsKnownComplete; ++i)
      markGroupComplete(i);
  }

  void checkPendingRead(memref::LoadOp load) {
    AccessRegion access{load.getMemRef(), SmallVector<Value>(load.getIndices()),
                        1, load};
    for (const PendingCopy &pending : copies) {
      if (pending.complete)
        continue;
      Overlap overlap = classifyOverlap(access, pending.destination);
      if (overlap == Overlap::Unknown) {
        load.emitRemark(
            "async-check: unknown whether this read overlaps an incomplete "
            "asynchronous copy");
        continue;
      }
      if (overlap != Overlap::Yes)
        continue;
      InFlightDiagnostic diagnostic = load.emitError(
          "async-copy destination is read before a sufficient wait");
      diagnostic.attachNote(pending.copy->getLoc())
          << "overlapping asynchronous copy was issued here";
      foundError = true;
    }
  }

  void checkPendingWrite(memref::StoreOp store) {
    AccessRegion access{store.getMemRef(),
                        SmallVector<Value>(store.getIndices()), 1, store};
    for (const PendingCopy &pending : copies) {
      if (pending.complete)
        continue;
      Overlap overlap = classifyOverlap(access, pending.destination);
      if (overlap == Overlap::Unknown) {
        store.emitRemark(
            "async-check: unknown whether this write overlaps an incomplete "
            "asynchronous copy");
        continue;
      }
      if (overlap != Overlap::Yes)
        continue;
      InFlightDiagnostic diagnostic = store.emitError(
          "shared-memory write overlaps an incomplete asynchronous copy");
      diagnostic.attachNote(pending.copy->getLoc())
          << "overlapping asynchronous copy was issued here";
      foundError = true;
    }
  }

  void checkSharedRead(memref::LoadOp load) {
    if (!hasWorkgroupMemorySpace(load.getMemRef()))
      return;
    AccessRegion access{load.getMemRef(), SmallVector<Value>(load.getIndices()),
                        1, load};
    for (const AccessRegion &store : storesSinceBarrier) {
      Overlap overlap = classifyOverlap(store, access);
      if (overlap == Overlap::No || isSameThreadOwnedAccess(store, access))
        continue;
      InFlightDiagnostic diagnostic = load.emitWarning(
          "possible cross-thread shared-memory RAW hazard; no recognized "
          "workgroup barrier separates this load from an earlier store");
      diagnostic.attachNote(store.origin->getLoc())
          << "possibly overlapping shared-memory store is here";
    }
  }

  void checkBlock(Block &block) {
    copies.clear();
    groups.clear();
    copyTokens.clear();
    groupTokens.clear();
    storesSinceBarrier.clear();

    for (Operation &operation : block) {
      if (auto copy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(operation)) {
        handleCopy(copy);
        continue;
      }
      if (auto group = dyn_cast<nvgpu::DeviceAsyncCreateGroupOp>(operation)) {
        handleGroup(group);
        continue;
      }
      if (auto wait = dyn_cast<nvgpu::DeviceAsyncWaitOp>(operation)) {
        handleWait(wait);
        continue;
      }
      if (auto barrier = dyn_cast<gpu::BarrierOp>(operation)) {
        if (hasProvenUniformParticipation(barrier))
          storesSinceBarrier.clear();
        continue;
      }
      if (auto load = dyn_cast<memref::LoadOp>(operation)) {
        checkPendingRead(load);
        checkSharedRead(load);
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(operation)) {
        checkPendingWrite(store);
        if (hasWorkgroupMemorySpace(store.getMemRef()))
          storesSinceBarrier.push_back(
              AccessRegion{store.getMemRef(),
                           SmallVector<Value>(store.getIndices()), 1, store});
      }
    }
  }

  bool foundError = false;
  SmallVector<PendingCopy> copies;
  SmallVector<CopyGroup> groups;
  DenseMap<Value, unsigned> copyTokens;
  DenseMap<Value, unsigned> groupTokens;
  SmallVector<AccessRegion> storesSinceBarrier;
};

class CheckAsyncCorrectnessPass
    : public PassWrapper<CheckAsyncCorrectnessPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckAsyncCorrectnessPass)

  StringRef getArgument() const final { return "check-async-correctness"; }

  StringRef getDescription() const final {
    return "Check supported GPU synchronization and asynchronous-copy rules";
  }

  void runOnOperation() override {
    bool foundError = false;
    getOperation().walk([&](gpu::GPUFuncOp function) {
      foundError |= checkDivergentBarriers(function);
    });

    StraightLineBlockChecker checker;
    foundError |= checker.check(getOperation().getBodyRegion());
    if (foundError)
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckAsyncCorrectnessPass() {
  return std::make_unique<CheckAsyncCorrectnessPass>();
}

void registerAsyncCheckPasses() {
  mlir::PassRegistration<CheckAsyncCorrectnessPass>();
}

} // namespace async_check
