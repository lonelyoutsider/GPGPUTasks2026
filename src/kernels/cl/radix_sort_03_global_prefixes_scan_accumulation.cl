#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#include "helpers/rassert.cl"
#include "../defines.h"

// Exclusive prefix scan of the flattened digit-major histogram.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sums,
    __global uint* global_prefix,
    uint count,
    uint pow2)
{
    const uint i = get_global_id(0);
    if (i >= count) return;
    const uint block_size = 1u << pow2;
    if (i & block_size)
        global_prefix[i] += pow2_sums[i / block_size - 1u];
}

// Each workgroup adds one scanned block total to a contiguous 1024-value tile.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scan_add_offsets(
    __global uint* prefix,
    __global const uint* block_prefix,
    uint count)
{
    const uint block = get_group_id(0);
    if (block == 0u) return;
    const uint base = block * RADIX_SCAN_BLOCK_SIZE + 4u * get_local_id(0);
    const uint offset = block_prefix[block];
    if (base + 3u < count) {
        vstore4(vload4(0, prefix + base) + offset, 0, prefix + base);
    } else {
        if (base < count) prefix[base] += offset;
        if (base + 1u < count) prefix[base + 1u] += offset;
        if (base + 2u < count) prefix[base + 2u] += offset;
    }
}
