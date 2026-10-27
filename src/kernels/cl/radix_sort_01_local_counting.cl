#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#include "helpers/rassert.cl"
#include "../defines.h"

// Stable 4-bit tile partition: a workgroup handles GROUP_SIZE elements.
// Block histograms are stored digit-major: counts[d * groups + g].
#define RADIX_BITS 4u
#define RADIX 16u
#define RADIX_MASK 15u

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* local_digit_rank,
    __global uint* block_counts,
    uint n,
    uint groups,
    uint shift)
{
    const uint lid = get_local_id(0);
    const uint g = get_group_id(0);
    const uint i = get_global_id(0);
    const uint tile_base = g * GROUP_SIZE;
    const uint valid_count = min((uint)GROUP_SIZE, n - tile_base);

    __local uint keys_a[GROUP_SIZE];
    __local uint keys_b[GROUP_SIZE];
    __local uint indices_a[GROUP_SIZE];
    __local uint indices_b[GROUP_SIZE];
    __local uint scan[GROUP_SIZE];
    __local uint starts[RADIX];
    __local uint counts[RADIX];

    // Invalid elements are MAX_UINT. Stable local partition keeps them at the
    // end of the highest digit bin, after all valid elements with digit 15.
    keys_a[lid] = (i < n) ? input[i] : 0xffffffffu;
    indices_a[lid] = lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint* src_keys = keys_a;
    __local uint* dst_keys = keys_b;
    __local uint* src_indices = indices_a;
    __local uint* dst_indices = indices_b;

    for (uint subbit = 0u; subbit < RADIX_BITS; ++subbit) {
        const uint b = (src_keys[lid] >> (shift + subbit)) & 1u;
        scan[lid] = b;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Inclusive scan of the current bit across a single workgroup.
        for (uint offset = 1u; offset < GROUP_SIZE; offset <<= 1u) {
            uint add = 0u;
            if (lid >= offset) add = scan[lid - offset];
            barrier(CLK_LOCAL_MEM_FENCE);
            scan[lid] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const uint ones_before = scan[lid] - b;
        const uint zeros_total = GROUP_SIZE - scan[GROUP_SIZE - 1u];
        const uint dst = b ? zeros_total + ones_before : lid - ones_before;
        dst_keys[dst] = src_keys[lid];
        dst_indices[dst] = src_indices[lid];
        barrier(CLK_LOCAL_MEM_FENCE);

        __local uint* tmp = src_keys;
        src_keys = dst_keys; dst_keys = tmp;
        tmp = src_indices;
        src_indices = dst_indices; dst_indices = tmp;
    }

    if (lid < RADIX) {
        starts[lid] = GROUP_SIZE;
        counts[lid] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint digit = (src_keys[lid] >> shift) & RADIX_MASK;
    if (lid == 0u || ((src_keys[lid - 1u] >> shift) & RADIX_MASK) != digit)
        starts[digit] = lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each digit has one last element in the locally sorted sequence.
    if (lid == GROUP_SIZE - 1u ||
        ((src_keys[lid + 1u] >> shift) & RADIX_MASK) != digit) {
        const uint end = min(lid + 1u, valid_count);
        counts[digit] = (end > starts[digit]) ? end - starts[digit] : 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX) block_counts[lid * groups + g] = counts[lid];

    const uint original_lid = src_indices[lid];
    if (tile_base + original_lid < n)
        local_digit_rank[tile_base + original_lid] = lid - starts[digit];
}

// Six packed histograms, three 9-bit counters per uint, cover all 16 bins.
// A counter is at most 256, so additions never carry into the adjacent field.
// One exclusive tree scan gives stable ranks directly in original input order;
// no key/index permutations or four sequential bit partitions are necessary.
#define PACKED_HISTOGRAMS 6u
#define PACKED_SCAN_STRIDE (GROUP_SIZE + GROUP_SIZE / 32)
#define PACKED_SCAN_INDEX(i) ((i) + ((i) >> 5u))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_local_counting_packed(
    __global const uint* input,
    __global uint* local_digit_rank,
    __global uint* block_counts,
    uint n,
    uint groups,
    uint shift)
{
    const uint lid = get_local_id(0);
    const uint i = get_global_id(0);
    const uint digit = i < n ? (input[i] >> shift) & 15u : 0u;
    const uint component = digit / 3u;
    const uint bit = 9u * (digit % 3u);
    const uint flag = i < n ? 1u << bit : 0u;
    __local uint scan[PACKED_HISTOGRAMS][PACKED_SCAN_STRIDE];
    __local uint totals[PACKED_HISTOGRAMS];

    #pragma unroll
    for (uint c = 0; c < PACKED_HISTOGRAMS; ++c)
        scan[c][PACKED_SCAN_INDEX(lid)] = c == component ? flag : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1u; stride < GROUP_SIZE; stride <<= 1u) {
        const uint right = (lid + 1u) * (stride << 1u) - 1u;
        if (right < GROUP_SIZE) {
            #pragma unroll
            for (uint c = 0; c < PACKED_HISTOGRAMS; ++c)
                scan[c][PACKED_SCAN_INDEX(right)] += scan[c][PACKED_SCAN_INDEX(right - stride)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < PACKED_HISTOGRAMS) {
        totals[lid] = scan[lid][PACKED_SCAN_INDEX(GROUP_SIZE - 1u)];
        scan[lid][PACKED_SCAN_INDEX(GROUP_SIZE - 1u)] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        const uint right = (lid + 1u) * (stride << 1u) - 1u;
        if (right < GROUP_SIZE) {
            #pragma unroll
            for (uint c = 0; c < PACKED_HISTOGRAMS; ++c) {
                const uint left = scan[c][PACKED_SCAN_INDEX(right - stride)];
                scan[c][PACKED_SCAN_INDEX(right - stride)] = scan[c][PACKED_SCAN_INDEX(right)];
                scan[c][PACKED_SCAN_INDEX(right)] += left;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n)
        local_digit_rank[i] = (scan[component][PACKED_SCAN_INDEX(lid)] >> bit) & 511u;
    if (lid < RADIX)
        block_counts[lid * groups + get_group_id(0)] =
            (totals[lid / 3u] >> (9u * (lid % 3u))) & 511u;
}
