/*
 * PRUNED LOW-Q ENUMERATOR
 *
 * Computes low-degree coefficients of P_n(q) via reverse Dumont generation
 * with branch-and-bound pruning.
 *
 * Optimizations:
 * 1. Iterative DFS (no recursion overhead)
 * 2. Bit-parallel color selection (jump to valid colors via CTZ)
 * 3. Pre-calculated constraint masks (Dumont inequalities)
 * 4. N*(N-1) parallel tasks for load balancing
 * 5. L1 cache locality (all state fits in registers/stack)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Tuning
int MAX_Q = 5;

// Global Stats
atomic<long long> g_perms_done{0};
atomic<long long> g_valid_pairs{0};
long long g_total_perms = 0;
chrono::high_resolution_clock::time_point g_last_report;

// Precomputed Masks for Dumont Constraints
// valid_first[k] = mask of c where k > 2c-1 (for first placement)
// valid_second[k] = mask of c where k < 2c   (for second placement)
uint32_t PRE_MASK_FIRST[64];  // indexed by k (1..2N)
uint32_t PRE_MASK_SECOND[64]; // indexed by k (1..2N)

void init_masks(int N) {
    for (int k = 1; k <= 2 * N; k++) {
        uint32_t m_first = 0;
        uint32_t m_second = 0;
        for (int c = 1; c <= N; c++) {
            // First placement rule: k > 2c - 1  => 2c < k + 1
            if (k > 2 * c - 1) m_first |= (1U << c);

            // Second placement rule: k < 2c
            if (k < 2 * c) m_second |= (1U << c);
        }
        PRE_MASK_FIRST[k] = m_first;
        PRE_MASK_SECOND[k] = m_second;
    }
}

template<int N>
struct ThreadResult {
    long long counts[32]; // Fixed size to avoid vector overhead
    long long valid_dumonts = 0;
    long long pruned_branches = 0;
    long long perms_checked = 0;

    ThreadResult() {
        memset(counts, 0, sizeof(counts));
    }
};

// ============================================================================
// ITERATIVE DUMONT SOLVER
// ============================================================================

template<int N>
inline void solve_dumont_iterative(const uint8_t* __restrict perm, ThreadResult<N>& res) {

    // Stack arrays (depth is 2N)
    int stack_q[2 * N + 2];
    uint32_t stack_mask[2 * N + 2];
    uint32_t stack_candidates[2 * N + 2];
    uint8_t stack_l2[2 * N + 2];

    // Track availability via bitmasks
    // avail_first: bit c is 1 if color c has NOT been used yet
    // avail_second: bit c is 1 if color c has been used EXACTLY ONCE
    uint32_t avail_first = (1U << (N + 1)) - 2; // Bits 1..N set
    uint32_t avail_second = 0;

    // Initial State
    int pos = 2 * N - 1;
    stack_q[pos] = 0;
    stack_mask[pos] = (1U << N) - 1; // Bits 0..N-1 set (for perm values)

    // Rule for Count=0 (Rightmost/p2): k > 2c-1.
    uint32_t valid_c = avail_first & PRE_MASK_FIRST[pos + 1];
    stack_candidates[pos] = valid_c;

    while (pos < 2 * N) {
        // 1. Check if we have candidates left at this level
        uint32_t c_mask = stack_candidates[pos];

        if (c_mask == 0) {
            // BACKTRACK
            pos++;
            if (pos >= 2 * N) return;

            int c = stack_l2[pos];

            if ((avail_second & (1U << c)) || (avail_first & (1U << c))) {
                avail_second &= ~(1U << c);
                avail_first |= (1U << c);
            } else {
                avail_second |= (1U << c);
            }

            continue;
        }

        // 2. Pick next candidate (CTZ is fast!)
        int c = __builtin_ctz(c_mask);
        stack_candidates[pos] &= ~(1U << c);

        stack_l2[pos] = c;

        // 4. Update Global State (avail masks)
        bool is_first_placement = (avail_first & (1U << c));

        if (is_first_placement) {
            avail_first &= ~(1U << c);
            avail_second |= (1U << c);
        } else {
            avail_second &= ~(1U << c);
        }

        // 5. Compute Q and Prune
        int current_q = stack_q[pos];
        uint32_t current_mask = stack_mask[pos];
        int next_q = current_q;
        uint32_t next_mask = current_mask;

        int row = pos / 2;
        int v = perm[c - 1];

        if (pos % 2 == 1) {
            // Odd Pos (Rightmost element of row)
            next_mask &= ~(1U << v);
        } else {
            // Even Pos (Leftmost element of row)
            int bottom_v = perm[row];

            next_q += __builtin_popcount(next_mask >> (bottom_v + 1));

            if (next_q >= MAX_Q) {
                // PRUNE!
                if (is_first_placement) {
                    avail_second &= ~(1U << c);
                    avail_first |= (1U << c);
                } else {
                    avail_second |= (1U << c);
                }
                res.pruned_branches++;
                continue;
            }

            next_mask |= (1U << bottom_v);
            next_mask &= ~(1U << v);
        }

        // 6. Descend
        if (pos == 0) {
            // BASE CASE: Found one!
            res.counts[next_q]++;
            res.valid_dumonts++;

            if (is_first_placement) {
                avail_second &= ~(1U << c);
                avail_first |= (1U << c);
            } else {
                avail_second |= (1U << c);
            }
        } else {
            // Setup next level
            int next_pos = pos - 1;
            stack_q[next_pos] = next_q;
            stack_mask[next_pos] = next_mask;

            // GENERATE CANDIDATES for next_pos
            int k = next_pos + 1;
            uint32_t candidates = 0;

            uint32_t c1 = avail_first & PRE_MASK_FIRST[k];
            uint32_t c2 = avail_second & PRE_MASK_SECOND[k];

            candidates = c1 | c2;

            // Rule: Boundary l2[2i-1] < l2[2i]
            if (next_pos % 2 == 1 && next_pos < 2 * N - 1) {
                int limit = stack_l2[pos];
                uint32_t limit_mask = (1U << limit) - 1;
                candidates &= limit_mask;
            }

            stack_candidates[next_pos] = candidates;
            pos = next_pos;
        }
    }
}

// ============================================================================
// PARALLEL DRIVER
// ============================================================================

template<int N>
void solve_parallel(int num_threads) {
    auto t0 = chrono::high_resolution_clock::now();
    g_perms_done = 0;
    g_valid_pairs = 0;
    g_last_report = chrono::high_resolution_clock::now();

    // Init lookup tables
    init_masks(N);

    // Calculate total permutations
    long long fact = 1;
    for (int i = 2; i <= N; i++) fact *= i;
    g_total_perms = fact;

    cout << "  Strategy: Super-Optimized Iterative DFS with Bit-Parallel Selection" << endl;
    cout << "  Iterating " << N << "! = " << fact << " perms" << endl;
    cout << "  Threads: " << num_threads << " | Tasks: " << N*(N-1) << " | Pruning q >= " << MAX_Q << endl;

    ThreadResult<N> global_res;

    // Parallelize first TWO levels: N * (N-1) tasks
    int total_tasks = N > 1 ? N * (N - 1) : N;

    #pragma omp parallel num_threads(num_threads)
    {
        ThreadResult<N> local_res;
        uint8_t p[N];

        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < total_tasks; ++task) {
            int a = task / (N - 1);
            int rem = task % (N - 1);
            int b = rem;
            if (b >= a) b++;

            p[0] = a;
            p[1] = b;

            uint32_t used = (1U << a) | (1U << b);

            auto generate_rest = [&](auto&& self, int k, uint32_t m) -> void {
                if (k == N) {
                    solve_dumont_iterative<N>(p, local_res);

                    local_res.perms_checked++;
                    long long done = ++g_perms_done;
                    if (done % 500000 == 0) {
                        auto now = chrono::high_resolution_clock::now();
                        double elapsed = chrono::duration<double>(now - g_last_report).count();
                        if (elapsed > 0.5) {
                            g_last_report = now;
                            double pct = 100.0 * done / g_total_perms;
                            #pragma omp critical
                            {
                                cout << "\r  Progress: " << fixed << setprecision(1) << pct << "% ("
                                     << done/1000000 << "M/" << g_total_perms/1000000 << "M perms)" << flush;
                            }
                        }
                    }
                    return;
                }

                uint32_t avail = ((1U << N) - 1) & ~m;
                while (avail) {
                    int idx = __builtin_ctz(avail);
                    p[k] = idx;
                    self(self, k + 1, m | (1U << idx));
                    avail &= ~(1U << idx);
                }
            };

            if (N > 1) {
                generate_rest(generate_rest, 2, used);
            } else {
                solve_dumont_iterative<N>(p, local_res);
            }
        }

        #pragma omp critical
        {
            for (int q = 0; q < MAX_Q; ++q) global_res.counts[q] += local_res.counts[q];
            global_res.valid_dumonts += local_res.valid_dumonts;
            global_res.pruned_branches += local_res.pruned_branches;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    cout << "\r  Done.                                              " << endl;

    cout << endl << string(60, '=') << endl;
    cout << "n = " << N << endl;
    cout << "Time:            " << chrono::duration<double>(t1-t0).count() << "s" << endl;
    cout << "Valid pairs:     " << global_res.valid_dumonts << endl;
    cout << "Pruned branches: " << global_res.pruned_branches << endl;
    cout << string(60, '=') << endl;

    for (int q = 0; q < MAX_Q; ++q) {
        cout << "  [q^" << q << "] = " << global_res.counts[q] << endl;
    }
}

void print_help(const char* prog) {
    cout << "Pruned Low-Q Enumerator" << endl;
    cout << "Computes low-degree coefficients of P_n(q) via reverse Dumont generation.\n" << endl;
    cout << "Usage: " << prog << " <n> [max_q]" << endl;
    cout << "  n      : Triangle size (required, 5-16)" << endl;
    cout << "  max_q  : Track coefficients q^0 to q^(max_q-1) (default: 5)" << endl;
    cout << "\nExamples:" << endl;
    cout << "  " << prog << " 10      # n=10, track q^0..q^4" << endl;
    cout << "  " << prog << " 11 8    # n=11, track q^0..q^7" << endl;
    cout << "\nEnvironment:" << endl;
    cout << "  OMP_NUM_THREADS=6 " << prog << " 10   # Use 6 threads" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || string(argv[1]) == "-h" || string(argv[1]) == "--help") {
        print_help(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    int n = stoi(argv[1]);
    MAX_Q = (argc > 2) ? stoi(argv[2]) : 5;

    int threads = 1;
    #ifdef _OPENMP
    threads = omp_get_max_threads();
    #endif

    cout << "Super-Optimized Reverse Genocchi Solver" << endl;
    cout << string(60, '=') << endl;
    cout << "n = " << n << ", " << threads << " threads, MAX_Q = " << MAX_Q << endl;
    cout << string(60, '=') << endl << endl;

    switch(n) {
        case 5: solve_parallel<5>(threads); break;
        case 6: solve_parallel<6>(threads); break;
        case 7: solve_parallel<7>(threads); break;
        case 8: solve_parallel<8>(threads); break;
        case 9: solve_parallel<9>(threads); break;
        case 10: solve_parallel<10>(threads); break;
        case 11: solve_parallel<11>(threads); break;
        case 12: solve_parallel<12>(threads); break;
        case 13: solve_parallel<13>(threads); break;
        case 14: solve_parallel<14>(threads); break;
        case 15: solve_parallel<15>(threads); break;
        case 16: solve_parallel<16>(threads); break;
        default: cerr << "Support only n=5..16" << endl; return 1;
    }
    return 0;
}
