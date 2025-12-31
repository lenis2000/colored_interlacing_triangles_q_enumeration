/*
 * OPTIMIZED PRUNED LOW-Q ENUMERATOR
 * Author: Leo Petrov (Refactored for High-Performance)
 *
 * Optimizations:
 * 1. Greedy-Fail Pruning: Iterates values Largest->Smallest.
 * Since cost(v) is monotonic decreasing with v, if v fails,
 * all v' < v also fail. Allows O(1) bulk pruning.
 * 2. Fine-Grained Parallelism: Tasks generated at Depth 3.
 * 3. Intrinsic Bit Manipulation: CLZ/LZCNT for fast iteration.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <cstring>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Tuning Constants
int MAX_Q = 5;
int INV_THRESHOLD = 10;

// Global Metrics
atomic<long long> g_perms_done{0};
atomic<long long> g_perms_skipped{0};
atomic<long long> g_tasks_done{0};
long long g_total_tasks = 0;
chrono::high_resolution_clock::time_point g_start_time;
atomic<double> g_last_print{0};

// Precomputed masks
uint32_t PRE_MASK_FIRST[64];
uint32_t PRE_MASK_SECOND[64];

void init_masks(int N) {
    for (int k = 1; k <= 2 * N; k++) {
        uint32_t m_first = 0, m_second = 0;
        for (int c = 1; c <= N; c++) {
            if (k > 2 * c - 1) m_first |= (1U << c);
            if (k < 2 * c) m_second |= (1U << c);
        }
        PRE_MASK_FIRST[k] = m_first;
        PRE_MASK_SECOND[k] = m_second;
    }
}

template<int N>
struct ThreadResult {
    long long counts[32];
    long long valid_dumonts = 0;
    long long pruned_branches = 0;
    long long perms_checked = 0;
    long long perms_skipped = 0;
    char padding[64]; // Prevent false sharing between threads

    ThreadResult() { memset(counts, 0, sizeof(counts)); }
};

// --- Dumont Solver (Scalar Optimized) ---
template<int N>
inline void solve_dumont_iterative(const uint8_t* __restrict perm, ThreadResult<N>& res) {
    // Stack-allocated tight arrays
    int stack_q[2 * N + 2];
    uint32_t stack_mask[2 * N + 2];
    uint32_t stack_candidates[2 * N + 2];
    uint8_t stack_l2[2 * N + 2];

    uint32_t avail_first = (1U << (N + 1)) - 2;
    uint32_t avail_second = 0;

    int pos = 2 * N - 1;
    stack_q[pos] = 0;
    stack_mask[pos] = (1U << N) - 1;
    stack_candidates[pos] = avail_first & PRE_MASK_FIRST[pos + 1];

    // Hot loop
    while (pos < 2 * N) {
        uint32_t c_mask = stack_candidates[pos];

        if (c_mask == 0) {
            pos++;
            if (pos >= 2 * N) return;
            // Backtrack logic
            int c = stack_l2[pos];
            // Toggle avail bit back
            if ((avail_second & (1U << c)) || (avail_first & (1U << c))) {
                avail_second &= ~(1U << c);
                avail_first |= (1U << c);
            } else {
                avail_second |= (1U << c);
            }
            continue;
        }

        // Pick smallest available candidate (Dumont rule standard)
        int c = __builtin_ctz(c_mask);
        stack_candidates[pos] &= ~(1U << c); // Remove from current level
        stack_l2[pos] = c;

        // Update avail sets
        bool is_first_placement = (avail_first & (1U << c));
        if (is_first_placement) {
            avail_first &= ~(1U << c);
            avail_second |= (1U << c);
        } else {
            avail_second &= ~(1U << c);
        }

        int current_q = stack_q[pos];
        uint32_t current_mask = stack_mask[pos];
        int next_q = current_q;
        uint32_t next_mask = current_mask;

        // Calculate Q impact
        if (pos % 2 == 1) {
            int v = perm[c - 1]; // Top row access
            next_mask &= ~(1U << v);
        } else {
            int row = pos / 2;
            int v = perm[c - 1];
            int bottom_v = perm[row];

            // Inversion-like Q update
            next_q += __builtin_popcount(next_mask >> (bottom_v + 1));

            // Q-Pruning
            if (next_q >= MAX_Q) {
                // Revert avail state immediately
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

        if (pos == 0) {
            // Leaf reached
            res.counts[next_q]++;
            res.valid_dumonts++;
            // Revert state for next iteration
            if (is_first_placement) {
                avail_second &= ~(1U << c);
                avail_first |= (1U << c);
            } else {
                avail_second |= (1U << c);
            }
        } else {
            // Recurse down
            int next_pos = pos - 1;
            stack_q[next_pos] = next_q;
            stack_mask[next_pos] = next_mask;

            int k = next_pos + 1;
            uint32_t cand = (avail_first & PRE_MASK_FIRST[k]) | (avail_second & PRE_MASK_SECOND[k]);

            // Triangle bound constraint
            if (next_pos % 2 == 1 && next_pos < 2 * N - 1) {
                int limit = stack_l2[pos];
                cand &= (1U << limit) - 1;
            }

            stack_candidates[next_pos] = cand;
            pos = next_pos;
        }
    }
}

// --- Recursive Permutation Generator (Reverse Iteration) ---
template<int N>
void generate_perms_recursive(
    uint8_t* __restrict p,
    int pos,
    uint32_t avail,
    uint32_t placed,
    int inv,
    ThreadResult<N>& res
) {
    if (pos == N) {
        solve_dumont_iterative<N>(p, res);
        res.perms_checked++;
        return;
    }

    uint32_t remaining = avail;

    // OPTIMIZATION: Reverse Iteration (Largest Value to Smallest Value)
    // Why: popcount(placed >> (v+1)) is monotonic non-increasing with v.
    //      Largest v has LOWEST cost. Smallest v has HIGHEST cost.
    //      If we check largest v first and it works, great.
    //      If we hit a v where cost is too high, ALL smaller v are also too high.
    //      We can break immediately.

    while (remaining) {
        // Find largest available value (MSB)
        int v = 31 - __builtin_clz(remaining);

        // Calculate inversions
        int new_inv = __builtin_popcount(placed >> (v + 1));
        int total_inv = inv + new_inv;

        if (total_inv > INV_THRESHOLD) {
            // Pruning Logic:
            // Since we iterate v downwards (High->Low), and Cost increases as v decreases,
            // if current v fails, all remaining (smaller) values in 'remaining' will also fail.
            // We prune them ALL in one go.

            // remaining currently includes v and all smaller available candidates.
            int skipped_count = __builtin_popcount(remaining);

            long long subfact = 1;
            // Precomputing factorials for small N is cheap, or loop
            for (int i = 2; i <= N - pos - 1; i++) subfact *= i;

            res.perms_skipped += skipped_count * subfact;
            // No continue needed; we are done with this node.
            break;
        }

        // Valid branch, recurse
        remaining ^= (1U << v); // Remove v from local iteration set
        p[pos] = v;

        generate_perms_recursive<N>(
            p, pos + 1,
            avail ^ (1U << v),   // remove v from passed available
            placed | (1U << v),  // add v to placed
            total_inv,
            res
        );
    }
}

template<int N>
void solve_parallel(int num_threads) {
    auto t0 = chrono::high_resolution_clock::now();
    g_start_time = t0;
    g_perms_done = 0;
    g_perms_skipped = 0;
    g_tasks_done = 0;
    g_last_print = 0;

    init_masks(N);

    // Calculate total perms for display
    long long fact = 1;
    for (int i = 2; i <= N; i++) fact *= i;

    // Determine optimal task depth based on N
    int task_depth = (N >= 8) ? 4 : 3;
    if (N >= 12) task_depth = 5;

    cout << "  Strategy: Recursive DFS + Greedy Reverse-Order Pruning" << endl;
    cout << "  Threads: " << num_threads << " | Depth-" << task_depth << " Tasks" << endl;

    struct Task {
        uint8_t p[5];  // Up to depth 5
        uint32_t placed;
        uint32_t avail;
        int inv;
        int depth;
    };

    vector<Task> tasks;

    // Recursive task generator with early pruning
    uint32_t full_mask = (1U << N) - 1;

    function<void(Task&, int)> gen_tasks = [&](Task& t, int pos) {
        if (pos == task_depth || pos == N) {
            tasks.push_back(t);
            t.depth = pos;
            return;
        }

        // Iterate largest to smallest for consistent pruning
        for (int v = N - 1; v >= 0; --v) {
            if (!(t.avail & (1U << v))) continue;

            int new_inv = __builtin_popcount(t.placed >> (v + 1));
            int total_inv = t.inv + new_inv;

            if (total_inv > INV_THRESHOLD) break; // All smaller will fail too

            Task nt = t;
            nt.p[pos] = v;
            nt.placed |= (1U << v);
            nt.avail &= ~(1U << v);
            nt.inv = total_inv;
            nt.depth = pos + 1;

            gen_tasks(nt, pos + 1);
        }
    };

    Task root;
    root.placed = 0;
    root.avail = full_mask;
    root.inv = 0;
    root.depth = 0;
    gen_tasks(root, 0);

    g_total_tasks = tasks.size();
    cout << "  Generated " << g_total_tasks << " initial tasks (Depth " << task_depth << ")." << endl;

    ThreadResult<N> global_res;

    #pragma omp parallel num_threads(num_threads)
    {
        ThreadResult<N> local_res;
        uint8_t p[N]; // Thread-local permutation buffer

        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < tasks.size(); ++i) {
            // Load task
            const auto& t = tasks[i];
            for (int j = 0; j < t.depth; ++j) p[j] = t.p[j];

            // Run solver from task depth
            if (t.depth < N) {
                generate_perms_recursive<N>(p, t.depth, t.avail, t.placed, t.inv, local_res);
            } else {
                // Complete permutation
                solve_dumont_iterative<N>(p, local_res);
                local_res.perms_checked++;
            }

            // Time-based progress (every ~1s)
            long long done = ++g_tasks_done;
            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - g_start_time).count();
            double last = g_last_print.load();
            if (elapsed - last >= 1.0 || done == g_total_tasks) {
                if (g_last_print.compare_exchange_weak(last, elapsed)) {
                    cout << "\r  " << done << "/" << g_total_tasks << " tasks"
                         << " (" << (100 * done / g_total_tasks) << "%)"
                         << " | " << fixed << setprecision(1) << elapsed << "s " << flush;
                }
            }
        }

        // Reduction
        #pragma omp critical
        {
            global_res.perms_checked += local_res.perms_checked;
            global_res.perms_skipped += local_res.perms_skipped;
            for (int q = 0; q < MAX_Q; ++q) global_res.counts[q] += local_res.counts[q];
            global_res.valid_dumonts += local_res.valid_dumonts;
            global_res.pruned_branches += local_res.pruned_branches;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    cout << "  Done." << endl;
    cout << string(60, '=') << endl;
    cout << "n = " << N << endl;
    cout << "Time:              " << chrono::duration<double>(t1-t0).count() << "s" << endl;
    cout << "Perms evaluated:   " << global_res.perms_checked << endl;
    cout << "Perms pruned:      " << global_res.perms_skipped << " (inv > " << INV_THRESHOLD << ")" << endl;
    cout << "Valid pairs:       " << global_res.valid_dumonts << endl;
    cout << "Dumont branches pruned: " << global_res.pruned_branches << endl;
    cout << string(60, '=') << endl;

    for (int q = 0; q < MAX_Q; ++q) {
        cout << "  [q^" << q << "] = " << global_res.counts[q] << endl;
    }
}

// Wrapper to handle templates
void run_solver(int n, int threads) {
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
        default: cerr << "Support n=5..16" << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <n> [max_q] [inv_threshold]" << endl;
        return 1;
    }
    int n = stoi(argv[1]);
    MAX_Q = (argc > 2) ? stoi(argv[2]) : 5;
    INV_THRESHOLD = (argc > 3) ? stoi(argv[3]) : 10;

    int threads = 1;
    #ifdef _OPENMP
    threads = omp_get_max_threads();
    #endif

    run_solver(n, threads);
    return 0;
}
