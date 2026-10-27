#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

// Four contiguous values per work-item; shared by host and OpenCL kernels.
#define RADIX_SCAN_ITEMS_PER_THREAD 4
#define RADIX_SCAN_BLOCK_SIZE (GROUP_SIZE * RADIX_SCAN_ITEMS_PER_THREAD)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
