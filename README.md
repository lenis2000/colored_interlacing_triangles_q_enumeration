# q-Polynomial Enumeration for Colored Interlacing Triangles

Compute the q-counting polynomial $T_2(n;q)$ for colored interlacing triangles of depth $N=2$.

## Programs

| Program | Use Case | Description |
|---------|----------|-------------|
| `enumerate_triangles` | Full polynomial | Computes all coefficients of P_n(q) for small n |
| `enumerate_prune` | Low-q coefficients | Dumont pruning for large n |
| `enumerate_prune_inv` | Low-q (fastest CPU) | Inversion heuristic + Dumont pruning |
| `enumerate_gpu` | Low-q (GPU) | Metal GPU acceleration (macOS only) |

## Quick Start

```bash
make                              # Build all programs
./enumerate_triangles 6           # Compute full P_6(q)
./enumerate_prune 10 5            # Compute q^0..q^4 of P_10(q)
OMP_NUM_THREADS=6 ./enumerate_prune_inv 12 5 10   # Fastest CPU version
./enumerate_gpu 10 5 10           # GPU version (macOS)
```

Requires OpenMP. On macOS: `brew install libomp` and `brew install gcc`

## enumerate_triangles

Full polynomial computation for small n (≤ 9).

```bash
./enumerate_triangles <max_n>   # Compute P_n(q) for n up to max_n
```

### Output Files

| File | Description |
|------|-------------|
| `T2_n<N>_poly.dat` | Polynomial coefficients (q-power, coefficient pairs) |
| `T2_n<N>_psi.dat` | Per-triangle psi statistic distributions |

Data files for n ≤ 8 are committed to git. For n = 9, the psi file is too large:
- **T2_n9_psi.dat**: https://storage.lpetrov.cc/qGenocchi/T2_n9_psi.dat

### Polynomial File Format (`T2_n<N>_poly.dat`)

```
# T_2(6;q) polynomial data
# Format: q_power coefficient
0 1
1 20
2 245
...
```

### Psi Data File Format (`T2_n<N>_psi.dat`)

Each line contains one canonical triangle and its psi distribution over all n! permutations:

```
# Format: triangle_index l1[0..n-1] l2[0..2n-1] | psi_dist (q:count pairs)
idx l1_0 l1_1 ... l1_{n-1} | l2_0 l2_1 ... l2_{2n-1} | q0:c0 q1:c1 ...
```

Example for n=6:
```
0 0 1 2 3 4 5 | 0 1 2 4 5 0 3 1 2 3 4 5 | 4:9 5:58 6:132 7:161 8:161 9:132 10:58 11:9
```

Colors are 0-indexed. The psi distribution shows how many of the 720 permutations yield each q-power.

### Algorithm

1. **Phase 1 (Canonical enumeration)**:
   - Generate all valid l2 middle permutations via Dumont derangement bijection
   - Check interlacing condition (each color appears in pattern: l2, l1, l2)
   - Check boundary ordering constraint
   - Parallel processing with batched permutation generation

2. **Phase 2 (q-power computation)**:
   - For each canonical triangle and each of n! permutations
   - Compute q-power using bitmask and `__builtin_popcount` (O(n) per pair)
   - ILP x4: Process 4 permutations simultaneously for cache efficiency
   - Thread-local counting with OpenMP parallelization

## enumerate_prune

Fast low-q coefficient computation for large n (10-16) using reverse Dumont generation with branch-and-bound pruning.

```bash
./enumerate_prune <n> [max_q]
  n      : Triangle size (required, 5-16)
  max_q  : Track coefficients q^0 to q^(max_q-1) (default: 5)

# Examples:
OMP_NUM_THREADS=6 ./enumerate_prune 10     # n=10, track q^0..q^4
OMP_NUM_THREADS=6 ./enumerate_prune 11 8   # n=11, track q^0..q^7
```

### Algorithm: Reverse Dumont Generation with Pruning

Rather than generating all canonical triangles (Dumont derangements) and iterating permutations, we **invert the loops**:

1. **Outer loop**: Iterate over all n! permutations (parallelized into N*(N-1) tasks)
2. **Inner loop**: Generate Dumont derangements on-the-fly via iterative DFS
3. **Pruning**: When partial ψ-value reaches max_q, prune entire branch

### Optimizations

1. **Iterative DFS**: No recursion overhead; all state on explicit stack arrays
2. **Bit-parallel color selection**: Pre-calculated constraint masks for Dumont inequalities; jump to valid colors via CTZ (`__builtin_ctz`)
3. **N*(N-1) parallel tasks**: Better load balancing than N tasks (first two permutation elements fixed per task)
4. **L1 cache locality**: All state fits in registers/stack; zero heap allocation during search

### Why This Works for Low q

The key insight is the asymmetry between the two spaces:
- **Permutations**: n! grows "slowly" (e.g., 10! ≈ 3.6M, 11! ≈ 40M)
- **Dumont derangements**: H_n grows exponentially faster (e.g., H_10 ≈ 7.8B, H_11 ≈ 392B)

In the standard approach, we must enumerate all H_n derangements. In the pruned approach, we never materialize this space. Instead, for each permutation, we build Dumont derangements iteratively while tracking the ψ-statistic. The moment ψ reaches the threshold max_q, we **prune the entire subtree**.

### Pruning Efficiency

For small max_q (e.g., 5), the ψ-value accumulates quickly during iteration:
- Most branches are pruned at depth 3-5 (out of 2n positions)
- Pruning eliminates >99.99% of the Dumont space H_n
- Only O(polynomial in n) valid pairs survive per permutation

This makes it feasible to compute low-degree coefficients for n ≤ 16, where full enumeration would require examining 10^14+ triangle-permutation pairs.

### Memory Usage

Zero heap allocation during search:
- Permutation state: O(n) on stack
- Iterative DFS state: O(n) stack arrays for q, mask, candidates, colors
- Pre-computed constraint masks: O(n) global arrays

## enumerate_prune_inv

**Fastest CPU version** with inversion-based early pruning heuristic.

```bash
./enumerate_prune_inv <n> [max_q] [inv_threshold]
  n              : Triangle size (required, 5-16)
  max_q          : Track coefficients q^0 to q^(max_q-1) (default: 5)
  inv_threshold  : Skip permutations with >threshold inversions (default: 10)

# Examples:
OMP_NUM_THREADS=6 ./enumerate_prune_inv 10 5 10   # ~27s for n=10
OMP_NUM_THREADS=6 ./enumerate_prune_inv 12 5 10   # n=12, low-q coefficients
```

### Inversion Heuristic

Key insight: permutations with many inversions tend to produce high ψ-values. By filtering permutations with inv(π) > threshold **before** running the Dumont solver, we skip most of the search space while still finding all low-q contributions.

For `max_q=5` and `inv_threshold=10`:
- n=10: Evaluates only 51,909 of 3,628,800 permutations (1.4%)
- Speedup: ~3-4x over `enumerate_prune`

### Additional Optimizations

1. **Greedy-fail iteration**: Values tested largest→smallest; if v fails the inversion bound, all smaller values also fail (bulk pruning)
2. **Adaptive task depth**: Depth-4 parallelism for n≥8, depth-5 for n≥12
3. **Time-based progress**: Reports progress every ~1 second without thread contention

## enumerate_gpu

**GPU-accelerated version** using Apple Metal (macOS only).

```bash
./enumerate_gpu <n> [max_q] [inv_threshold]

# Examples:
./enumerate_gpu 10 5 10    # ~5-7s for n=10 on M2 Pro
./enumerate_gpu 12 5 10    # GPU acceleration for larger n
```

### Performance

| n | CPU (enumerate_prune_inv) | GPU (enumerate_gpu) | Speedup |
|---|---------------------------|---------------------|---------|
| 10 | ~27s | ~5-7s | ~4-5x |

### Implementation Details

1. **CPU generates permutations** with inv ≤ threshold
2. **GPU runs Dumont solver** for each permutation in parallel
3. **Chunked dispatch**: 16k threads per Metal dispatch (hardware limit)
4. **Atomic reduction**: Thread-safe q-coefficient accumulation

### Requirements

- macOS with Apple Silicon or AMD GPU
- Metal framework (included in Xcode)

## Symmetry Reductions

1. **S_n symmetry**: Fix level1 = (0,1,...,n-1), enumerate only "canonical" triangles
2. **Boundary ordering**: Only count triangles with l2[2i-1] < l2[2i], reducing by factor 2^{n-1}
   - The n-1 independent involutions swapping these pairs preserve the ψ-statistic

## Result Format

The polynomial is output as:
```
T_2(n;q) = 2^{n-1} * P_n(q)
```

where P_n(q) is palindromic of degree n(n-1)/2.

## Building

```bash
make clean && make
```

The Makefile auto-detects macOS vs Linux for OpenMP configuration.
