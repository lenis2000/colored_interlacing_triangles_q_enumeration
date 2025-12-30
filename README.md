# q-Polynomial Enumeration for Colored Interlacing Triangles

Compute the q-counting polynomial $T_2(n;q)$ for colored interlacing triangles of depth $N=2$.

See also [https://github.com/lenis2000/colored_interlacing_triangles_enumeration](https://github.com/lenis2000/colored_interlacing_triangles_enumeration) for plain enumeration for any depth.

arXiv paper: (TO BE INSTERTED)


## Quick Start

```bash
make                    # Build with OpenMP support
./enumerate_triangles 6 # Compute T_2(6;q)
```

Requires OpenMP. On macOS: `brew install libomp`

## Output Files

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

## Algorithm

### Symmetry Reductions

1. **S_n symmetry**: Fix level1 = (0,1,...,n-1), enumerate only "canonical" triangles
2. **Boundary ordering**: Only count triangles with l2[2i-1] < l2[2i], reducing by factor 2^{n-1}

### Implementation

1. **Phase 1 (Canonical enumeration)**:
   - Generate all valid l2 middle permutations
   - Check interlacing condition (each color appears in pattern: l2, l1, l2)
   - Check boundary ordering constraint
   - Parallel processing with batched permutation generation

2. **Phase 2 (q-power computation)**:
   - For each canonical triangle and each of n! permutations
   - Compute q-power using bitmask and __builtin_popcount (O(n) per pair)
   - ILP x4: Process 4 permutations simultaneously for cache efficiency
   - Thread-local counting with OpenMP parallelization

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
