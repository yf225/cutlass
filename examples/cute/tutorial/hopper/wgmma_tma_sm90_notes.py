#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include <torch/torch.h>

"""
NOTE(yf225): how to implement CUTLASS C++ -> PICA conversion:
It should be driven by LLM, but verified with our PICA -> CUTLASS C++ convertor,
to ensure that the PICA code can be used to generate back the exact same CUTLASS C++ code (after C++ formatting).
The LLM-driven C++ -> PICA conversion process should be as follows:
1. Use C++ AST parser or LLM to chunk the C++ code into smaller chunks (e.g. functions and classes / structs)
2. For each chunk, use LLM to generate the PICA code.
3. Verify the generated PICA code by using the PICA -> CUTLASS C++ convertor to check if the generated PICA code can be used to
generate back the exact same CUTLASS C++ code. If they don't match, tell the mismatch error to LLM and ask it to fix it.
4. Repeat the process until the generated PICA code can be used to generate back the exact same CUTLASS C++ code.

NOTE(yf225): how C++ namespace is handled in PICA
1. Translation from C++ to Python
Case 1:
```
namespace cute {
...
}
```
=> 
```
#[[namespace]] cute {
...
#} [[end namespace]]
```

Case 2:
`#include <cute/tensor.hpp>` => Must specify what's being used from the header file, e.g. `from cute.tensor import X`
During the CUTLASS C++ -> PICA conversion, we should be able to know what's being used from the header file
by searching unknown symbols through all the PICA-managed `#include`-ed header files
(we always convert starting from the leaf header files, so for downstream files, we should only need to do symbol searching
within already-converted PICA-managed header files which are Python files (so easier to AST parse and find symbols)).

Case 3:
`using namespace cute;` => PICA-managed files will not have this line - it will be generated during C++ codegen by looking up what namespace each symbol in the file belongs to.
Specifically, we will maintain a list of symbols under each namespace (written within cute/__namespace__.py, cutlass/__namespace__.py, cute/detail/__namespace__.py, etc. files).
These __namespace__.py files are checked in to the repo (not dynamically generated), and will be used to
do symbol lookup and populate the `using namespace X` line during C++ codegen.
Q: How are __namespace__.py files generated?
A: We will have an additional "populate" step after any PICA-managed files are changed (as an offline step before committing the changes),
which walks through the changed file to detect new additions to the namespace, and
add the new symbols to the corresponding __namespace__.py files.
"""

# using namespace cute;

# template <class ElementA,
#           class ElementB,
#           class SmemLayoutA,  // (M,K,P)
#           class SmemLayoutB>  // (N,K,P)
# struct SharedStorage
# {
#   alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
#   alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

#   uint64_t tma_barrier[size<2>(SmemLayoutA{})];
#   uint64_t mma_barrier[size<2>(SmemLayoutA{})];
# };

class SharedStorage:
    def __new__(_, ElementA: cls, ElementB: cls, SmemLayoutA: cls, SmemLayoutB: cls):
        class _inner:
            A: alignas(128)[cute.ArrayEngine.spec(ElementA, cosize_v.spec(SmemLayoutA))]
            B: alignas(128)[cute.ArrayEngine.spec(ElementB, cosize_v.spec(SmemLayoutB))]

            tma_barrier: [uint64_t] * size.spec(2)(SmemLayoutA())
            mma_barrier: [uint64_t] * size.spec(2)(SmemLayoutA())
        return _inner


# template <class ProblemShape, class CtaTiler,
#           class TA, class SmemLayoutA, class TmaA,
#           class TB, class SmemLayoutB, class TmaB,
#           class TC, class CStride, class TiledMma,
#           class Alpha, class Beta>
# __global__ static
# __launch_bounds__(decltype(size(TiledMma{}))::value)
# void
# gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
#             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
#             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
#             TC      * C, CStride dC, TiledMma mma,
#             Alpha alpha, Beta beta)
# {
def gemm_device(ProblemShape: cls, CtaTiler: cls,
    TA: cls, SmemLayoutA: cls, TmaA: cls,
    TB: cls, SmemLayoutB: cls, TmaB: cls,
    TC: cls, CStride: cls, TiledMma: cls,
    Alpha: cls, Beta: cls):

    @attrs(___global___, static, __launch_bounds__(decltype(size(TiledMma())).value))
    def _inner(
        shape_MNK: ProblemShape,
        cta_tiler: CtaTiler,
        A: ConstPtr[TA],
        tma_a: CUTLASS_GRID_CONSTANT[Const[TmaA]]
        B: ConstPtr[TB],
        tma_b: CUTLASS_GRID_CONSTANT[Const[TmaB]]
        C: Ptr[TC],
        dC: CStride,
        mma: TiledMma,
        alpha: Alpha,
        beta: Beta) -> None:
    # NOTE(yf225): these are all static asserts and will only enforced during codegen, and will not show up in the generated code.

#   // Preconditions
#   CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
#   CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

        CUTE_STATIC_ASSERT_V(rank(shape_MNK) == 3)                   # (M, N, K)
        CUTE_STATIC_ASSERT_V(rank(cta_tiler) == 3)                   # (BLK_M, BLK_N, BLK_K)
  
#   static_assert(is_static<SmemLayoutA>::value);
#   static_assert(is_static<SmemLayoutB>::value);

    # # TODO(yf225): these two are in the @extra_types(...), for users to explicitly specify.
    # static_assert_template_arg_exists("SmemLayoutA")
    # static_assert_template_arg_exists("SmemLayoutB")

#   CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
#   CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
#   CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
#   CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

        CUTE_STATIC_ASSERT_V(SmemLayoutA().shape[0] == cta_tiler.shape[0])  # BLK_M
        CUTE_STATIC_ASSERT_V(SmemLayoutB().shape[0] == cta_tiler.shape[1])  # BLK_N
        CUTE_STATIC_ASSERT_V(SmemLayoutA().shape[1] == cta_tiler.shape[2])  # BLK_K
        CUTE_STATIC_ASSERT_V(SmemLayoutB().shape[1] == cta_tiler.shape[2])  # BLK_K


#   CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

        CUTE_STATIC_ASSERT_V(congruent(make_shape(shape_MNK, 0, 1), dC))         # dC strides for shape MN

#   //
#   // Full and Tiled Tensors
#   //

#   // Represent the full tensors
#   auto [M, N, K] = shape_MNK;
#   Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
#   Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
#   Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

        M, N, K = shape_MNK
        mA = tma_a.get_tma_tensor(make_shape(M,K))                   # (M,K) TMA Tensor
        mB = tma_b.get_tma_tensor(make_shape(N,K))                   # (N,K) TMA
        mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC)      # (M,N)

#   // Get the appropriate blocks for this thread block
#   auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
#   Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
#   Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
#   Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

        # TODO(yf225): where are _ and _1 and X defined? can we simplify this?
        cta_coord = make_coord(blockIdx.x, blockIdx.y, _)              # (m,n,k)
        gA = local_tile(mA, cta_tiler, cta_coord, Step(_1, X, _1)())  # (BLK_M,BLK_K,k)
        gB = local_tile(mB, cta_tiler, cta_coord, Step(X, _1, _1)())  # (BLK_N,BLK_K,k)
        gC = local_tile(mC, cta_tiler, cta_coord, Step(_1, _1, X)())  # (BLK_M,BLK_N)

#   // Shared memory tensors
#   extern __shared__ char shared_memory[];
#   using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
#   SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
#   Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
#   Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

        shared_memory: List["extern", "__shared__", "char"] = []
        SharedStorage = SharedStorage(TA, TB, SmemLayoutA, SmemLayoutB)
        smem: Ref[SharedStorage] = reinterpret_cast(Ptr[SharedStorage])(shared_memory) # reinterpret_cast<SharedStorage*>(shared_memory)
        sA: Tensor = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA()) # (BLK_M,BLK_K,PIPE)
        sB: Tensor = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB()) # (BLK_N,BLK_K,PIPE)


#   //
#   // Partition the copying of A and B tiles
#   //
#   // TUTORIAL:
#   //   These are TMA partitionings, which have a dedicated custom partitioner.
#   //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
#   //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
#   //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
#   //     with the understanding that the TMA is responsible for everything in mode-0.
#   //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
#   //

#   auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
#                                     group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

#   auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
#                                     group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

#   // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
#   constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
#                                       + sizeof(make_tensor_like(tensor<0>(tBsB)));

        tAgA, tAsA = tma_partition(tma_a, Int.spec(0)(), Layout.spec(_1)(),
                                group_modes.spec(0, 2)(sA), group_modes.spec(0, 2)(gA))  # (TMA,k) and (TMA,PIPE)

        tBgB, tBsB = tma_partition(tma_b, Int.spec(0)(), Layout.spec(_1)(),
                                group_modes.spec(0, 2)(sB), group_modes.spec(0, 2)(gB))  # (TMA,k) and (TMA,PIPE)

        # The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
        tma_transaction_bytes: ConstExpr[int] = sizeof(make_tensor_like(tensor.spec(0)(tAsA))) + sizeof(make_tensor_like(tensor.spec(0)(tBsB)))

#   //
#   // PREFETCH
#   //

#   auto K_PIPE_MAX = size<1>(tAsA);

        K_PIPE_MAX = size.spec(1)(tAsA)

#   // Total count of tiles
#   int k_tile_count = size<1>(tAgA);
#   // Current tile index in gmem to read from
#   int k_tile = 0;

        k_tile_count = size.spec(1)(tAgA)
        k_tile = 0

#   // Initialize Barriers
#   int warp_idx = cutlass::canonical_warp_idx_sync();
#   int lane_predicate = cute::elect_one_sync();
#   uint64_t* producer_mbar = smem.tma_barrier;
#   uint64_t* consumer_mbar = smem.mma_barrier;

#   using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
#   using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
#   CUTE_UNROLL
#   for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
#     if ((warp_idx == 0) && lane_predicate) {
#       ProducerBarType::init(&producer_mbar[pipe],   1);
#       ConsumerBarType::init(&consumer_mbar[pipe], 128);
#     }
#   }

        warp_idx: int = cutlass.canonical_warp_idx_sync()
        lane_predicate: int = cute.elect_one_sync()
        producer_mbar: Ptr[uint64_t] = smem.tma_barrier
        consumer_mbar: Ptr[uint64_t] = smem.mma_barrier

        ProducerBarType = cutlass.arch.ClusterTransactionBarrier
        ConsumerBarType = cutlass.arch.ClusterBarrier
        #[[CUTE_UNROLL]]
        for pipe in range(K_PIPE_MAX):
            if (warp_idx == 0) and lane_predicate:
                ProducerBarType.init(addr_of(producer_mbar[pipe]), 1)
                ConsumerBarType.init(addr_of(consumer_mbar[pipe]), 128)

#   // Ensure barrier init is complete on all CTAs
#   cluster_sync();

        cluster_sync()

#   // Start async loads for all pipes
#   CUTE_UNROLL
#   for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
#   {
#     if ((warp_idx == 0) && lane_predicate)
#     {
#       // Set expected Tx Bytes after each reset / init
#       ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
#       copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
#       copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
#     }
#     --k_tile_count;
#     ++k_tile;
#   }

        #[[CUTE_UNROLL]]
        for pipe in range(K_PIPE_MAX):
            if (warp_idx == 0) and lane_predicate:
                ProducerBarType.arrive_and_expect_tx(addr_of(producer_mbar[pipe]), tma_transaction_bytes)
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe))
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe))
            k_tile_count -= 1
            k_tile += 1

#   //
#   // Define A/B partitioning and C accumulators
#   //
#   // TUTORIAL:
#   //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
#   //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
#   //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
#   //     there is no need for copy(tCsA, tCrA) in the mainloop.
#   //

#   ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
#   Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
#   Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
#   Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

#   // Allocate accumulators and clear them
#   Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
#   clear(tCrC);

#   // Allocate "fragments"
#   Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
#   Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

        thr_mma: ThrMMA = mma.get_thread_slice(threadIdx.x)
        tCsA: Tensor = thr_mma.partition_A(sA)                               # (MMA,MMA_M,MMA_K,PIPE)
        tCsB: Tensor = thr_mma.partition_B(sB)                               # (MMA,MMA_N,MMA_K,PIPE)
        tCgC: Tensor = thr_mma.partition_C(gC)                               # (MMA,MMA_M,MMA_N)

        tCrC: Tensor = thr_mma.make_fragment_C(tCgC)                         # (MMA,MMA_M,MMA_N)
        clear(tCrC)

        tCrA: Tensor = thr_mma.make_fragment_A(tCsA)                         # (MMA,MMA_M,MMA_K,PIPE)
        tCrB: Tensor = thr_mma.make_fragment_B(tCsB)                         # (MMA,MMA_N,MMA_K,PIPE)


#   //
#   // PIPELINED MAIN LOOP
#   //
#   // TUTORIAL:
#   //   Rather than interleaving the stages and instructions like in SM70 and SM80,
#   //     the SM90 mainloops rely on explicit producer-consumer synchronization
#   //     on the purely async instructions TMA and MMA.
#   //   More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops.
#   //

#   // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
#   //   that flips each cycle through K_PIPE_MAX.
#   auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
#   auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

#   CUTE_NO_UNROLL
#   while (k_tile_count > -K_PIPE_MAX)
#   {
#     // Wait for Producer to complete
#     int read_pipe = read_state.index();
#     ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

#     // MMAs to cover 1 K_TILE
#     warpgroup_arrive();
#     gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
#     warpgroup_commit_batch();

#     // Wait for all MMAs in a K_TILE to complete
#     warpgroup_wait<0>();

#     // Notify that consumption is done
#     ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
#     ++read_state;

#     if ((warp_idx == 0) && lane_predicate)
#     {
#       int pipe = write_state.index();
#       // Wait for Consumer to complete consumption
#       ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
#       // Set expected Tx Bytes after each reset / init
#       ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
#       copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
#       copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
#       ++write_state;
#     }
#     --k_tile_count;
#     ++k_tile;
#   }

        write_state = cutlass.PipelineState[K_PIPE_MAX]()             # TMA writes
        read_state = cutlass.PipelineState[K_PIPE_MAX]()               # MMA  reads

        #[[CUTE_NO_UNROLL]]
        while k_tile_count > -K_PIPE_MAX:
            # Wait for Producer to complete
            read_pipe = read_state.index()
            ProducerBarType.wait(addr_of(producer_mbar[read_pipe]), read_state.phase())

            # MMAs to cover 1 K_TILE
            warpgroup_arrive()
            gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC)     # (V,M) x (V,N) => (V,M,N)
            warpgroup_commit_batch()

            # Wait for all MMAs in a K_TILE to complete
            warpgroup_wait(0)

            # Notify that consumption is done
            ConsumerBarType.arrive(addr_of(consumer_mbar[read_pipe]))
            read_state += 1

            if (warp_idx == 0) and lane_predicate:
                pipe = write_state.index()
                # Wait for Consumer to complete consumption
                ConsumerBarType.wait(addr_of(consumer_mbar[pipe]), write_state.phase())
                # Set expected Tx Bytes after each reset / init
                ProducerBarType.arrive_and_expect_tx(addr_of(producer_mbar[pipe]), tma_transaction_bytes)
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe))
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe))
                write_state += 1
            k_tile_count -= 1
            k_tile += 1

#   //
#   // Epilogue (unpredicated)
#   //

#   axpby(alpha, tCrC, beta, tCgC);

        axpby(alpha, tCrC, beta, tCgC)

    return _inner

# }

# // Setup params for an NT GEMM
# template <class TA, class TB, class TC,
#           class Alpha, class Beta>
# void
# gemm_nt(int m, int n, int k,
#         Alpha alpha,
#         TA const* A, int ldA,
#         TB const* B, int ldB,
#         Beta beta,
#         TC      * C, int ldC,
#         cudaStream_t stream = 0)
# {

@template(TA, TB, TC, Alpha, Beta)
def gemm_nt(
    m: int,
    n: int,
    k: int,
    alpha: Alpha,
    A: ConstPtr[TA],
    ldA: int,
    B: ConstPtr[TB],
    ldB: int,
    beta: Beta,
    C: Ptr[TC],
    ldC: int,
    stream: cudaStream_t = 0) -> None:

#   // Define shapes (dynamic)
#   auto M = int(m);
#   auto N = int(n);
#   auto K = int(k);
#   auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    M = int(m)
    N = int(n)
    K = int(k)
    prob_shape = make_shape(M, N, K)                     # (M, N, K)

#   // Define TN strides (mixed)
#   auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
#   auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
#   auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

    dA = make_stride(Int.spec(1)(), ldA)                      # (dM, dK)
    dB = make_stride(Int.spec(1)(), ldB)                      # (dN, dK)
    dC = make_stride(Int.spec(1)(), ldC)                      # (dM, dN)

#   // Define CTA tile sizes (static)
#   auto bM = Int<128>{};
#   auto bN = Int<128>{};
#   auto bK = Int< 64>{};
#   auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
#   auto bP = Int<  3>{};  // Pipeline

    bM = Int.spec(128)()
    bN = Int.spec(128)()
    bK = Int.spec(64)()
    cta_tiler = make_shape(bM, bN, bK)                   # (BLK_M, BLK_N, BLK_K)
    bP = Int.spec(3)()  # Pipeline

#   // Define the smem layouts (static)
#   auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
#   auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

    sA = tile_to_shape(GMMA.Layout_MN_SW128_Atom.spec(TA)(), make_shape(bM,bK,bP))
    sB = tile_to_shape(GMMA.Layout_MN_SW128_Atom.spec(TB)(), make_shape(bN,bK,bP))

#   // Define the MMA
#   TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

    tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS.spec(GMMA.Major.MN, GMMA.Major.MN)())

#   // Define the TMAs
#   // Create Global memory tensors for TMA inspection
#   Tensor mA = make_tensor(A, make_shape(M,K), dA);
#   Tensor mB = make_tensor(B, make_shape(N,K), dB);

    mA = make_tensor(A, make_shape(M,K), dA)
    mB = make_tensor(B, make_shape(N,K), dB)

#   // Create TMA Atoms with the desired copy operation on the source and destination
#   Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
#   Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

    tmaA: Copy_Atom = make_tma_atom(SM90_TMA_LOAD(), mA, sA(_,_,0), make_shape(bM,bK))
    tmaB: Copy_Atom = make_tma_atom(SM90_TMA_LOAD(), mB, sB(_,_,0), make_shape(bN,bK))

#   //
#   // Setup and Launch
#   //

#   // Launch parameter setup
#   int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
#   dim3 dimBlock(size(tiled_mma));
#   dim3 dimCluster(2, 1, 1);
#   dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
#                round_up(size(ceil_div(n, bN)), dimCluster.y));
#   cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

#   void const* kernel_ptr = reinterpret_cast<void const*>(
#                               &gemm_device<decltype(prob_shape), decltype(cta_tiler),
#                                            TA, decltype(sA), decltype(tmaA),
#                                            TB, decltype(sB), decltype(tmaB),
#                                            TC, decltype(dC), decltype(tiled_mma),
#                                            decltype(alpha), decltype(beta)>);

#   CUTE_CHECK_ERROR(cudaFuncSetAttribute(
#     kernel_ptr,
#     cudaFuncAttributeMaxDynamicSharedMemorySize,
#     smem_size));

    smem_size = int(sizeof(SharedStorage.spec(TA, TB, decltype(sA), decltype(sB))()))
    dimBlock = dim3(size(tiled_mma))
    dimCluster = dim3(2, 1, 1)
    dimGrid = dim3(round_up(size(ceil_div(m, bM)), dimCluster.x),
                         round_up(size(ceil_div(n, bN)), dimCluster.y))
    params = cutlass.ClusterLaunchParams(dimGrid, dimBlock, dimCluster, smem_size)

    kernel_ptr = reinterpret_cast.spec(ConstPtr[None])(
        addr_of(gemm_device.spec(decltype(prob_shape), decltype(cta_tiler),
                                 TA, decltype(sA), decltype(tmaA),
                                 TB, decltype(sB), decltype(tmaB),
                                 TC, decltype(dC), decltype(tiled_mma),
                                 decltype(Alpha), decltype(Beta))))
    
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size))

#   // Kernel Launch
#   cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
#                                                              prob_shape, cta_tiler,
#                                                              A, tmaA,
#                                                              B, tmaB,
#                                                              C, dC, tiled_mma,
#                                                              alpha, beta);
#   CUTE_CHECK_LAST();

#   if (status != cutlass::Status::kSuccess) {
#     std::cerr << "Error: Failed at kernel Launch" << std::endl;
#   }

    status = cutlass.launch_kernel_on_cluster(params, kernel_ptr,
                                              prob_shape, cta_tiler,
                                              A, tmaA,
                                              B, tmaB,
                                              C, dC, tiled_mma,
                                              alpha, beta)
    CUTE_CHECK_LAST()

    if status != cutlass.Status.kSuccess:
        print("Error: Failed at kernel Launch")
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  int  smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

  auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                  TA, decltype(sA), decltype(tmaA),
                                  TB, decltype(sB), decltype(tmaB),
                                  TC, decltype(dC), decltype(tiled_mma),
                                  decltype(alpha), decltype(beta)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}
