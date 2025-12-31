/*
 * GPU-ACCELERATED LOW-Q ENUMERATOR (Metal)
 *
 * Strategy:
 * 1. CPU generates all valid permutations (inv <= threshold)
 * 2. GPU runs Dumont solver for each permutation in parallel
 * 3. Atomic reduction of q-counts
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

int MAX_Q = 5;
int INV_THRESHOLD = 10;

// Generate all permutations with inv <= threshold
void generate_permutations(int N, vector<uint8_t>& perms, int& perm_count) {
    vector<uint8_t> p(N);
    uint32_t full_mask = (1U << N) - 1;

    function<void(int, uint32_t, uint32_t, int)> gen = [&](int pos, uint32_t avail, uint32_t placed, int inv) {
        if (pos == N) {
            for (int i = 0; i < N; i++) perms.push_back(p[i]);
            perm_count++;
            return;
        }

        for (int v = N - 1; v >= 0; --v) {
            if (!(avail & (1U << v))) continue;

            int new_inv = __builtin_popcount(placed >> (v + 1));
            int total_inv = inv + new_inv;

            if (total_inv > INV_THRESHOLD) break;

            p[pos] = v;
            gen(pos + 1, avail & ~(1U << v), placed | (1U << v), total_inv);
        }
    };

    gen(0, full_mask, 0, 0);
}

// Metal shader source - Dumont solver for N=10
// DEBUG VERSION: Adding atomic counters to trace execution
const char* metalShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

constant short NMAX = 32;
constant short STACK_SIZE = 65; // 2*NMAX+1

kernel void dumont_solver(
    device const uchar* perms [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    device atomic_uint* valid_count [[buffer(2)]],
    constant short& N [[buffer(3)]],
    constant short& MAX_Q [[buffer(4)]],
    device atomic_uint* debug [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Load permutation into local array
    uchar perm[NMAX];
    for (short i = 0; i < N; i++) {
        perm[i] = perms[tid * N + i];
    }

    // Precompute Dumont constraint masks (uint for N<=32)
    uint PRE_MASK_FIRST[STACK_SIZE];
    uint PRE_MASK_SECOND[STACK_SIZE];
    for (short k = 1; k <= 2 * N; k++) {
        uint m_first = 0, m_second = 0;
        for (short c = 1; c <= N; c++) {
            if (k > 2 * c - 1) m_first |= (1u << c);
            if (k < 2 * c) m_second |= (1u << c);
        }
        PRE_MASK_FIRST[k] = m_first;
        PRE_MASK_SECOND[k] = m_second;
    }

    // Stack-based DFS state
    uchar stack_q[STACK_SIZE];       // Q values are small (0-31)
    uint stack_mask[STACK_SIZE];     // N<=32 bits needed
    uint stack_candidates[STACK_SIZE];
    uchar stack_l2[STACK_SIZE];

    // Available colors: first placement (bits 1..N) and second placement
    uint avail_first = (1u << (N + 1)) - 2u;  // bits 1..N set
    uint avail_second = 0u;

    // Start at position 2N-1 (rightmost of bottom row)
    short pos = 2 * N - 1;
    stack_q[pos] = 0;
    stack_mask[pos] = (1u << N) - 1u;  // bits 0..N-1 for perm values
    stack_candidates[pos] = avail_first & PRE_MASK_FIRST[pos + 1];

    while (pos < 2 * N) {
        uint c_mask = stack_candidates[pos];

        if (c_mask == 0u) {
            pos++;
            if (pos >= 2 * N) break;

            short c = stack_l2[pos];
            uint bit = 1u << c;
            if ((avail_second & bit) != 0u) {
                avail_second &= ~bit;
                avail_first |= bit;
            } else if ((avail_first & bit) != 0u) {
                avail_second &= ~bit;
                avail_first |= bit;
            } else {
                avail_second |= bit;
            }
            continue;
        }

        short c = ctz(c_mask);
        uint c_bit = 1u << c;
        stack_candidates[pos] &= ~c_bit;
        stack_l2[pos] = (uchar)c;

        bool is_first = (avail_first & c_bit) != 0u;
        if (is_first) {
            avail_first &= ~c_bit;
            avail_second |= c_bit;
        } else {
            avail_second &= ~c_bit;
        }

        short next_q = stack_q[pos];
        uint next_mask = stack_mask[pos];

        short v = perm[c - 1];
        uint v_bit = 1u << v;

        if (pos % 2 == 1) {
            next_mask &= ~v_bit;
        } else {
            short row = pos / 2;
            short bottom_v = perm[row];

            next_q += popcount(next_mask >> (bottom_v + 1));

            if (next_q >= MAX_Q) {
                if (is_first) {
                    avail_second &= ~c_bit;
                    avail_first |= c_bit;
                } else {
                    avail_second |= c_bit;
                }
                continue;
            }

            next_mask |= (1u << bottom_v);
            next_mask &= ~v_bit;
        }

        if (pos == 0) {
            atomic_fetch_add_explicit(&debug[2], 1u, memory_order_relaxed);
            if (next_q < MAX_Q) {
                atomic_fetch_add_explicit(&counts[next_q], 1u, memory_order_relaxed);
            }
            atomic_fetch_add_explicit(valid_count, 1u, memory_order_relaxed);

            if (is_first) {
                avail_second &= ~c_bit;
                avail_first |= c_bit;
            } else {
                avail_second |= c_bit;
            }
        } else {
            short next_pos = pos - 1;
            stack_q[next_pos] = next_q;
            stack_mask[next_pos] = next_mask;

            short k = next_pos + 1;
            uint cand = (avail_first & PRE_MASK_FIRST[k]) | (avail_second & PRE_MASK_SECOND[k]);

            if ((next_pos % 2 == 1) && (next_pos < 2 * N - 1)) {
                short limit = stack_l2[pos];
                cand &= (1u << limit) - 1u;
            }

            stack_candidates[next_pos] = cand;
            pos = next_pos;
        }
    }
}
)";

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <n> [max_q] [inv_threshold]" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    MAX_Q = (argc > 2) ? stoi(argv[2]) : 5;
    INV_THRESHOLD = (argc > 3) ? stoi(argv[3]) : 10;

    cout << "GPU Genocchi Solver (Metal)" << endl;
    cout << string(60, '=') << endl;
    cout << "n = " << N << ", MAX_Q = " << MAX_Q << ", INV_THRESHOLD = " << INV_THRESHOLD << endl;

    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        cerr << "Metal not available" << endl;
        return 1;
    }
    cout << "Device: " << [device.name UTF8String] << endl;

    // Create command queue
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Compile shader
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:metalShaderSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        cerr << "Shader compile error: " << [[error localizedDescription] UTF8String] << endl;
        return 1;
    }

    id<MTLFunction> kernel = [library newFunctionWithName:@"dumont_solver"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
    if (!pipeline) {
        cerr << "Pipeline error: " << [[error localizedDescription] UTF8String] << endl;
        return 1;
    }

    // Generate permutations on CPU
    cout << "Generating permutations..." << flush;
    auto t0 = chrono::high_resolution_clock::now();

    vector<uint8_t> perms;
    int perm_count = 0;
    generate_permutations(N, perms, perm_count);

    auto t1 = chrono::high_resolution_clock::now();
    cout << " " << perm_count << " perms in "
         << chrono::duration<double>(t1-t0).count() << "s" << endl;

    if (perm_count == 0) {
        cout << "No valid permutations!" << endl;
        return 0;
    }

    // Create buffers
    id<MTLBuffer> permBuffer = [device newBufferWithBytes:perms.data()
                                                   length:perms.size()
                                                  options:MTLResourceStorageModeShared];

    vector<uint32_t> counts(MAX_Q, 0);
    id<MTLBuffer> countBuffer = [device newBufferWithBytes:counts.data()
                                                    length:MAX_Q * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];

    uint32_t valid = 0;
    id<MTLBuffer> validBuffer = [device newBufferWithBytes:&valid
                                                    length:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];

    short N_short = (short)N;
    short MAX_Q_short = (short)MAX_Q;
    id<MTLBuffer> nBuffer = [device newBufferWithBytes:&N_short length:sizeof(short) options:MTLResourceStorageModeShared];
    id<MTLBuffer> maxqBuffer = [device newBufferWithBytes:&MAX_Q_short length:sizeof(short) options:MTLResourceStorageModeShared];

    // Debug buffer
    vector<uint32_t> debug_data(8, 0);
    id<MTLBuffer> debugBuffer = [device newBufferWithBytes:debug_data.data()
                                                     length:debug_data.size() * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];

    // Encode and run in chunks
    const int CHUNK_SIZE = 16384;  // Max threads per dispatch (stable for N<=32)
    cout << "Running GPU kernel in chunks of " << CHUNK_SIZE << "..." << endl;
    auto t2 = chrono::high_resolution_clock::now();

    NSUInteger threadGroupSize = min((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    int num_chunks = (perm_count + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * CHUNK_SIZE;
        int count = min(CHUNK_SIZE, perm_count - offset);

        // Create a sub-buffer view for this chunk's permutations
        id<MTLBuffer> chunkPermBuffer = [device newBufferWithBytes:perms.data() + offset * N
                                                            length:count * N
                                                           options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:chunkPermBuffer offset:0 atIndex:0];
        [encoder setBuffer:countBuffer offset:0 atIndex:1];
        [encoder setBuffer:validBuffer offset:0 atIndex:2];
        [encoder setBuffer:nBuffer offset:0 atIndex:3];
        [encoder setBuffer:maxqBuffer offset:0 atIndex:4];
        [encoder setBuffer:debugBuffer offset:0 atIndex:5];

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        auto t_chunk = chrono::high_resolution_clock::now();
        cout << "  Chunk " << (chunk + 1) << "/" << num_chunks << " (" << count << " perms) @ "
             << chrono::duration<double>(t_chunk - t2).count() << "s" << endl;
    }

    auto t3 = chrono::high_resolution_clock::now();
    cout << " done in " << chrono::duration<double>(t3-t2).count() << "s" << endl;

    // Read results
    memcpy(counts.data(), countBuffer.contents, MAX_Q * sizeof(uint32_t));
    memcpy(&valid, validBuffer.contents, sizeof(uint32_t));
    memcpy(debug_data.data(), debugBuffer.contents, debug_data.size() * sizeof(uint32_t));

    cout << string(60, '=') << endl;
    cout << "n = " << N << endl;
    cout << "Total time:        " << chrono::duration<double>(t3-t0).count() << "s" << endl;
    cout << "GPU kernel time:   " << chrono::duration<double>(t3-t2).count() << "s" << endl;
    cout << "Perms evaluated:   " << perm_count << endl;
    cout << "Valid pairs:       " << valid << endl;
    cout << "Debug[2] (pos==0): " << debug_data[2] << endl;
    cout << string(60, '=') << endl;

    for (int q = 0; q < MAX_Q; ++q) {
        cout << "  [q^" << q << "] = " << counts[q] << endl;
    }

    return 0;
}
