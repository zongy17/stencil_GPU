#include <cuda.h>
#include "common.h"

__forceinline__ __device__ data_t warp_max(data_t val) {
    for(int i = 16; i >= 1; i = i >> 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

__forceinline__ __device__ data_t warp_sum(data_t val) {
    for(int i = 16; i >= 1; i = i >> 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

__global__ void check_answer_first_kernel(cptr_t ans0, cptr_t ans1, \
                            int nx, int ny, int nz, \
                            int halo_x, int halo_y, int halo_z, \
                            ptr_t norm_1_out, ptr_t norm_2_out, ptr_t norm_inf_out) {
    
    __shared__ data_t norm_1_shmem[1024], norm_2_shmem[1024], norm_inf_shmem[1024];
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int tx = threadIdx.x + blockDim.x * blockIdx.x;
    const int ty = threadIdx.y + blockDim.y * blockIdx.y;
    const int tz = threadIdx.z + blockDim.z * blockIdx.z;
    data_t norm_1 = 0.0, norm_2 = 0.0, norm_inf = 0.0;
    if(tx < nx && ty < ny && tz < nz) {
        const int ldx = nx + halo_x * 2;
        const int ldy = ny + halo_y * 2;
        const int x = tx + halo_x;
        const int y = ty + halo_y;
        const int z = tz + halo_z;
        data_t err = abs(ans0[INDEX(x, y, z, ldx, ldy)] - ans1[INDEX(x, y, z, ldx, ldy)]);
        norm_1 = err;
        norm_2 = err * err;
        norm_inf = err;
    }
    norm_1_shmem[tid] = norm_1;
    norm_2_shmem[tid] = norm_2;
    norm_inf_shmem[tid] = norm_inf;
    __syncthreads();
    for(int size = 512; size >= 32; size = size >> 1) {
        if(tid < size) {
            norm_1_shmem[tid] += norm_1_shmem[tid + size];
            norm_2_shmem[tid] += norm_2_shmem[tid + size];
            norm_inf_shmem[tid] = max(norm_inf_shmem[tid], norm_inf_shmem[tid + size]);
        }
        __syncthreads();
    }
    if(tid < 32) {
        norm_1 = warp_sum(norm_1_shmem[tid]);
        norm_2 = warp_sum(norm_2_shmem[tid]);
        norm_inf = warp_max(norm_inf_shmem[tid]);
    }
    
    if(tid == 0) {
        norm_1_out[bid] = norm_1;
        norm_2_out[bid] = norm_2;
        norm_inf_out[bid] = norm_inf;
    }
}

__global__ void check_answer_rest_kernel(int size, cptr_t norm_1_in, \
                            cptr_t norm_2_in, cptr_t norm_inf_in, \
                            ptr_t norm_1_out, ptr_t norm_2_out, ptr_t norm_inf_out) {
    __shared__ data_t norm_1_shmem[1024], norm_2_shmem[1024], norm_inf_shmem[1024];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    data_t norm_1 = 0.0, norm_2 = 0.0, norm_inf = 0.0;
    if(gid < size) {
        norm_1 = norm_1_in[gid];
        norm_2 = norm_2_in[gid];
        norm_inf = norm_inf_in[gid];
    }
    norm_1_shmem[tid] = norm_1;
    norm_2_shmem[tid] = norm_2;
    norm_inf_shmem[tid] = norm_inf;
    __syncthreads();
    for(int size = 512; size >= 32; size = size >> 1) {
        if(tid < size) {
            norm_1_shmem[tid] += norm_1_shmem[tid + size];
            norm_2_shmem[tid] += norm_2_shmem[tid + size];
            norm_inf_shmem[tid] = max(norm_inf_shmem[tid], norm_inf_shmem[tid + size]);
        }
        __syncthreads();
    }
    if(tid < 32) {
        norm_1 = warp_sum(norm_1_shmem[tid]);
        norm_2 = warp_sum(norm_2_shmem[tid]);
        norm_inf = warp_max(norm_inf_shmem[tid]);
    }
    
    if(tid == 0) {
        norm_1_out[bid] = norm_1;
        norm_2_out[bid] = norm_2;
        norm_inf_out[bid] = norm_inf;
    }
}

typedef struct {
    data_t norm_1, norm_2, norm_inf;
} check_result_t;


inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

check_result_t check_answer(cptr_t ans0, cptr_t ans1, const dist_grid_info_t *info) {
    check_result_t result = {1e+12, 1e+12, 1e+12};
    int nx = info->global_size_x;
    int ny = info->global_size_y;
    int nz = info->global_size_z;
    int halo_x = info->halo_size_x;
    int halo_y = info->halo_size_y;
    int halo_z = info->halo_size_z;
    int first_bx = ceiling(nx, 16);
    int first_by = ceiling(ny, 8);
    int first_bz = ceiling(nz, 8);
    int reduce1_size = first_bx * first_by * first_bz;
    int buff_size = reduce1_size + ceiling(reduce1_size, 1024);
    int size = reduce1_size;
    data_t norm_1, norm_2, norm_inf;
    ptr_t tmp, buffer, norm_1_a, norm_1_b, norm_2_a, norm_2_b, norm_inf_a, norm_inf_b;
    if(cudaMalloc(&buffer, buff_size*3*sizeof(data_t)) != cudaSuccess) {
        return result;
    }
    norm_1_a = buffer;
    norm_2_a = buffer + buff_size;
    norm_inf_a = buffer + buff_size * 2;
    norm_1_b = norm_1_a + reduce1_size;
    norm_2_b = norm_2_a + reduce1_size;
    norm_inf_b = norm_inf_a + reduce1_size;
    check_answer_first_kernel<<<\
        dim3(first_bx, first_by, first_bz), \
        dim3(16,8,8)>>>(ans0, ans1, nx, ny, nz, \
            halo_x, halo_y, halo_z, norm_1_a, norm_2_a, norm_inf_a);
    while(size > 1) {
        int reduce_size = ceiling(size, 1024);
        check_answer_rest_kernel<<<reduce_size, 1024>>>(size, \
            norm_1_a, norm_2_a, norm_inf_a, norm_1_b, norm_2_b, norm_inf_b);
        tmp = norm_1_a; norm_1_a = norm_1_b; norm_1_b = tmp;
        tmp = norm_2_a; norm_2_a = norm_2_b; norm_2_b = tmp;
        tmp = norm_inf_a; norm_inf_a = norm_inf_b; norm_inf_b = tmp;
        size = reduce_size;
    }
    cudaMemcpy(&norm_1, norm_1_a, sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm_2, norm_2_a, sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm_inf, norm_inf_a, sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaFree(buffer);
    result.norm_1 = norm_1 / (info->global_size_x * info->global_size_y * info->global_size_z);
    result.norm_2 = sqrt(norm_2 / (info->global_size_x * info->global_size_y * info->global_size_z));
    result.norm_inf = norm_inf;
    return result;
}