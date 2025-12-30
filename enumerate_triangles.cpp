/*
 * NUCLEAR OPTIMIZED q-POLYNOMIAL ENUMERATION (N <= 16)
 *
 * Optimizations:
 * 1. Template Specialization: N is a compile-time constant
 * 2. ILP x4: Process 4 permutations per loop iteration
 * 3. Bitmask instead of Fenwick tree: O(1) popcount
 * 4. Flat memory layout for cache efficiency
 * 5. Array-based counting instead of map in hot path
 */

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Operation: 0=TOP (remove from mask), 1=BOTTOM (query+add)
struct Op {
    uint8_t type;
    uint8_t idx;
};

// Result structure including per-triangle psi data
struct EnumResult {
    long long total_count;
    map<int, long long> poly;
    vector<vector<uint8_t>> triangles;      // Each triangle: [l1..., l2...]
    vector<map<int, long long>> psi_dists;  // Per-triangle psi distribution
};

// Template worker for fixed N
template <int N>
EnumResult solve_for_n(int num_threads, bool verbose) {

    // 1. Generate all N! permutations (flat array)
    vector<uint8_t> p(N);
    iota(p.begin(), p.end(), 0);

    long long fact = 1;
    for (int i = 1; i <= N; ++i) fact *= i;

    vector<uint8_t> all_perms;
    all_perms.reserve(fact * N);
    do {
        all_perms.insert(all_perms.end(), p.begin(), p.end());
    } while (next_permutation(p.begin(), p.end()));

    // 2. Generate Canonical Triangles with boundary ordering
    struct Canonical {
        uint8_t l1[N];
        uint8_t l2[2*N];
    };
    vector<Canonical> canonicals;

    vector<int> l1(N);
    iota(l1.begin(), l1.end(), 1);

    // Middle elements: 2 of each color, minus endpoints
    vector<int> mid;
    for (int c = 1; c <= N; ++c) {
        int cnt = 2 - (c == 1) - (c == N);
        for (int k = 0; k < cnt; ++k) mid.push_back(c);
    }
    sort(mid.begin(), mid.end());

    // Count middle permutations for progress
    long long mid_fact = 1;
    for (int i = 1; i <= (int)mid.size(); ++i) mid_fact *= i;
    map<int,int> mid_counts;
    for (int x : mid) mid_counts[x]++;
    for (auto [k,v] : mid_counts) for (int i = 1; i <= v; ++i) mid_fact /= i;

    int mid_size = (int)mid.size();
    const size_t BATCH_SIZE = 50000000;  // 50M per batch

    if (verbose) {
        cout << "  Checking " << mid_fact << " middle permutations";
        if (mid_fact > (long long)BATCH_SIZE) cout << " (batched, parallel)";
        cout << "..." << endl;
    }

    long long total_checked = 0;
    mutex canonicals_mtx;

    while (total_checked < mid_fact) {
        // Generate batch
        size_t batch_count = 0;
        vector<uint8_t> batch_flat;
        batch_flat.reserve(min((size_t)BATCH_SIZE, (size_t)(mid_fact - total_checked)) * mid_size);

        while (batch_count < BATCH_SIZE && total_checked + (long long)batch_count < mid_fact) {
            for (int x : mid) batch_flat.push_back((uint8_t)x);
            batch_count++;
            if (!next_permutation(mid.begin(), mid.end())) break;
        }

        size_t num_in_batch = batch_flat.size() / mid_size;

        // Process batch in parallel
        #pragma omp parallel
        {
            vector<Canonical> local_canonicals;

            #pragma omp for schedule(dynamic, 10000)
            for (size_t idx = 0; idx < num_in_batch; idx++) {
                const uint8_t* mid_ptr = batch_flat.data() + idx * mid_size;

                // Build l2
                int l2[2*N];
                l2[0] = 1;
                l2[2*N - 1] = N;
                for (int j = 0; j < mid_size; j++) l2[j + 1] = mid_ptr[j];

                // Interlacing check
                bool ok = true;
                for (int color = 1; color <= N && ok; ++color) {
                    int cnt = 0;
                    int levels[3];
                    for (int i = 0; i < N && cnt < 3; ++i) {
                        if (l2[2*i] == color) levels[cnt++] = 2;
                        if (l1[i] == color) levels[cnt++] = 1;
                        if (l2[2*i + 1] == color) levels[cnt++] = 2;
                    }
                    if (cnt != 3 || levels[0] != 2 || levels[1] != 1 || levels[2] != 2) {
                        ok = false;
                    }
                }

                if (ok) {
                    // Boundary ordering
                    bool bound_ok = true;
                    for (int i = 1; i < N; ++i) {
                        if (l2[2*i - 1] >= l2[2*i]) { bound_ok = false; break; }
                    }
                    if (bound_ok) {
                        Canonical c;
                        for (int i = 0; i < N; ++i) c.l1[i] = l1[i] - 1;
                        for (int i = 0; i < 2*N; ++i) c.l2[i] = l2[i] - 1;
                        local_canonicals.push_back(c);
                    }
                }
            }

            // Merge local results
            #pragma omp critical
            {
                canonicals.insert(canonicals.end(), local_canonicals.begin(), local_canonicals.end());
            }
        }

        total_checked += num_in_batch;

        if (verbose) {
            cout << "\r  Phase 1: " << fixed << setprecision(1)
                 << (100.0 * total_checked / mid_fact) << "% [found: " << canonicals.size() << "]" << flush;
        }
    }

    if (verbose) {
        cout << "\r  Phase 1: 100% - " << canonicals.size() << " canonicals found      " << endl;
    }

    if (verbose) {
        cout << "  Canonicals: " << canonicals.size() << " | Perms: " << fact << endl;
    }

    // 3. Build operation sequence (right to left: Top, Bottom, Top)
    vector<Op> ops;
    ops.reserve(3 * N);
    for (int i = N - 1; i >= 0; --i) {
        ops.push_back({0, (uint8_t)(2*i + 1)});  // Top
        ops.push_back({1, (uint8_t)i});          // Bottom
        ops.push_back({0, (uint8_t)(2*i)});      // Top
    }

    // 4. Per-triangle psi distributions
    const int MAX_Q = N * N + 10;
    size_t num_canonicals = canonicals.size();
    vector<vector<long long>> per_tri_psi(num_canonicals, vector<long long>(MAX_Q, 0));

    if (verbose) {
        cout << "  Phase 2: Processing " << num_canonicals << " × " << fact << " = "
             << (num_canonicals * fact) << " pairs..." << endl;
    }

    atomic<size_t> progress{0};
    size_t p2_milestone = max((size_t)1, num_canonicals / 100);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        const uint8_t* perm_base = all_perms.data();
        size_t n_perms = fact;

        #pragma omp for schedule(dynamic, 1)
        for (size_t c_idx = 0; c_idx < num_canonicals; ++c_idx) {
            const Canonical& tri = canonicals[c_idx];
            auto& my_psi = per_tri_psi[c_idx];

            // ILP x4: Process 4 permutations at once
            size_t p_idx = 0;
            for (; p_idx + 4 <= n_perms; p_idx += 4) {
                const uint8_t* p0 = perm_base + (p_idx + 0) * N;
                const uint8_t* p1 = perm_base + (p_idx + 1) * N;
                const uint8_t* p2 = perm_base + (p_idx + 2) * N;
                const uint8_t* p3 = perm_base + (p_idx + 3) * N;

                int q0 = 0; uint32_t m0 = (1U << N) - 1;
                int q1 = 0; uint32_t m1 = (1U << N) - 1;
                int q2 = 0; uint32_t m2 = (1U << N) - 1;
                int q3 = 0; uint32_t m3 = (1U << N) - 1;

                for (const auto& op : ops) {
                    uint8_t c_idx_local = (op.type == 1) ? tri.l1[op.idx] : tri.l2[op.idx];

                    int v0 = p0[c_idx_local];
                    int v1 = p1[c_idx_local];
                    int v2 = p2[c_idx_local];
                    int v3 = p3[c_idx_local];

                    if (op.type == 1) {  // Bottom: count higher, then add
                        q0 += __builtin_popcount(m0 >> (v0 + 1)); m0 |= (1U << v0);
                        q1 += __builtin_popcount(m1 >> (v1 + 1)); m1 |= (1U << v1);
                        q2 += __builtin_popcount(m2 >> (v2 + 1)); m2 |= (1U << v2);
                        q3 += __builtin_popcount(m3 >> (v3 + 1)); m3 |= (1U << v3);
                    } else {  // Top: remove
                        m0 &= ~(1U << v0);
                        m1 &= ~(1U << v1);
                        m2 &= ~(1U << v2);
                        m3 &= ~(1U << v3);
                    }
                }

                my_psi[q0]++;
                my_psi[q1]++;
                my_psi[q2]++;
                my_psi[q3]++;
            }

            // Cleanup remaining permutations
            for (; p_idx < n_perms; ++p_idx) {
                const uint8_t* perm = perm_base + p_idx * N;
                int q = 0;
                uint32_t m = (1U << N) - 1;

                for (const auto& op : ops) {
                    uint8_t c_idx_local = (op.type == 1) ? tri.l1[op.idx] : tri.l2[op.idx];
                    int v = perm[c_idx_local];
                    if (op.type == 1) {
                        q += __builtin_popcount(m >> (v + 1));
                        m |= (1U << v);
                    } else {
                        m &= ~(1U << v);
                    }
                }
                my_psi[q]++;
            }

            // Progress update
            size_t done = ++progress;
            if (verbose && tid == 0 && done % p2_milestone == 0) {
                cout << "\r  Phase 2: " << fixed << setprecision(1)
                     << (100.0 * done / num_canonicals) << "%" << flush;
            }
        }
    }

    if (verbose) {
        cout << "\r  Phase 2: 100%                    " << endl;
    }

    // Build result
    EnumResult result;
    result.total_count = 0;

    // Convert canonicals to flat format and build psi distributions
    result.triangles.reserve(num_canonicals);
    result.psi_dists.reserve(num_canonicals);

    for (size_t i = 0; i < num_canonicals; ++i) {
        // Store triangle as flat vector: [l1_0, ..., l1_{N-1}, l2_0, ..., l2_{2N-1}]
        vector<uint8_t> tri_data(3 * N);
        for (int j = 0; j < N; ++j) tri_data[j] = canonicals[i].l1[j];
        for (int j = 0; j < 2*N; ++j) tri_data[N + j] = canonicals[i].l2[j];
        result.triangles.push_back(std::move(tri_data));

        // Convert per-triangle psi counts to map
        map<int, long long> psi_map;
        for (int q = 0; q < MAX_Q; ++q) {
            if (per_tri_psi[i][q] > 0) {
                psi_map[q] = per_tri_psi[i][q];
                result.poly[q] += per_tri_psi[i][q];
                result.total_count += per_tri_psi[i][q];
            }
        }
        result.psi_dists.push_back(std::move(psi_map));
    }

    return result;
}

// Generate polynomial string
string poly_to_string(const map<int, long long>& poly) {
    if (poly.empty()) return "0";

    vector<pair<int, long long>> sorted(poly.begin(), poly.end());
    sort(sorted.begin(), sorted.end());

    ostringstream oss;
    bool first = true;
    for (const auto& [power, count] : sorted) {
        if (count == 0) continue;
        if (!first) oss << " + ";
        first = false;

        if (power == 0) oss << count;
        else if (power == 1) {
            if (count == 1) oss << "q";
            else oss << count << "*q";
        } else {
            if (count == 1) oss << "q^" << power;
            else oss << count << "*q^" << power;
        }
    }
    return oss.str();
}

int main(int argc, char* argv[]) {
    int n = (argc > 1) ? stoi(argv[1]) : 6;

    int threads = 1;
    #ifdef _OPENMP
    threads = omp_get_max_threads();
    #endif

    cout << "q-Polynomial Enumeration (Nuclear ILP x4 Optimized)" << endl;
    cout << string(55, '=') << endl;
    cout << "n = " << n << " colors, " << threads << " threads" << endl;
    cout << string(55, '=') << endl << endl;

    auto start = chrono::high_resolution_clock::now();

    EnumResult result;

    // Template dispatch
    switch (n) {
        case 4: result = solve_for_n<4>(threads, true); break;
        case 5: result = solve_for_n<5>(threads, true); break;
        case 6: result = solve_for_n<6>(threads, true); break;
        case 7: result = solve_for_n<7>(threads, true); break;
        case 8: result = solve_for_n<8>(threads, true); break;
        case 9: result = solve_for_n<9>(threads, true); break;
        case 10: result = solve_for_n<10>(threads, true); break;
        default:
            cerr << "n=" << n << " not supported (add case to main)" << endl;
            return 1;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_s = chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0;

    long long factor = 1LL << (n - 1);
    long long full_count = result.total_count * factor;

    cout << endl << string(55, '=') << endl;
    cout << "RESULT" << endl;
    cout << string(55, '=') << endl;
    cout << "T_2(" << n << ") = " << full_count << endl;
    cout << "Time: " << time_s << "s" << endl;
    cout << endl;
    cout << "T_2(" << n << ";q) = 2^" << (n - 1) << " * (" << poly_to_string(result.poly) << ")" << endl;

    // Write polynomial data file
    string poly_file = "T2_n" + to_string(n) + "_poly.dat";
    {
        ofstream out(poly_file);
        out << "# T_2(" << n << ";q) polynomial data" << endl;
        out << "# Generated in " << time_s << "s" << endl;
        out << "# Format: q_power coefficient" << endl;
        out << "# T_2(" << n << ") = " << full_count << endl;
        out << "# T_2(" << n << ";q) = 2^" << (n-1) << " * P_" << n << "(q)" << endl;
        out << "# P_" << n << "(q) coefficients below:" << endl;
        for (const auto& [power, coeff] : result.poly) {
            out << power << " " << coeff << endl;
        }
    }
    cout << "Saved polynomial to " << poly_file << endl;

    // Write per-triangle psi data file
    string psi_file = "T2_n" + to_string(n) + "_psi.dat";
    {
        ofstream out(psi_file);
        out << "# Per-triangle psi statistic data for n=" << n << endl;
        out << "# Format: triangle_index l1[0..n-1] l2[0..2n-1] | psi_dist (q:count pairs)" << endl;
        out << "# Canonical triangles with level1 = (0,1,...,n-1) and boundary ordering" << endl;
        out << "# Total triangles: " << result.triangles.size() << endl;
        out << "# Each row: idx l1_data l2_data | q0:c0 q1:c1 ..." << endl;
        out << "#" << endl;

        for (size_t i = 0; i < result.triangles.size(); ++i) {
            const auto& tri = result.triangles[i];
            const auto& psi = result.psi_dists[i];

            out << i << " ";
            // l1 data (first n elements)
            for (int j = 0; j < n; ++j) out << (int)tri[j] << " ";
            out << "| ";
            // l2 data (next 2n elements)
            for (int j = 0; j < 2*n; ++j) out << (int)tri[n + j] << " ";
            out << "| ";
            // psi distribution
            for (const auto& [q, cnt] : psi) {
                out << q << ":" << cnt << " ";
            }
            out << endl;
        }
    }
    cout << "Saved psi data to " << psi_file << endl;

    return 0;
}
