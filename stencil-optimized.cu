#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include "common.h"

const char* version_name = "Optimized version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}
void destroy_dist_grid(dist_grid_info_t *grid_info) {}

__global__ void stencil_7_naive_kernel_1step(cptr_t  in, ptr_t  out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;// 线程负责的数据在全局的坐标（但注意不含halo的offset！）
    int ty = threadIdx.y  + blockDim.y * blockIdx.y;
    int tz = threadIdx.z  + blockDim.z * blockIdx.z;
    int ldxx = blockDim.x + 2;// local_g的宽度
    int ldyy = blockDim.y + 2;

    extern __shared__ data_t _local_g[];
    ptr_t local_g = _local_g;

    if(tx < nx && ty < ny && tz < nz) {// 保证超出了计算范围的线程不用工作
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int offset_x, offset_y, offset_z;
        int x = tx + halo_x;// 线程负责的数据在全局数组中的坐标，需要加上halo作为offset
        int y = ty + halo_y;
        int z = tz + halo_z;
        int local_x = threadIdx.x + halo_x;// 线程负责的数据在local_g中的坐标，需要加上halo作为offset
        int local_y = threadIdx.y + halo_y;
        int local_z = threadIdx.z + halo_z;

        // local_g的内部区域拷贝
        local_g[INDEX(local_x, local_y, local_z, ldxx, ldyy)] = in[INDEX(x, y, z, ldx, ldy)];
        // 如果线程负责计算的数据在local_g中处于有效区域的边缘（与halo区紧挨着），则负责将halo数据做填充
        offset_x = (local_x == 1) ? -1 : (local_x == blockDim.x) ? 1 : 0;
        offset_y = (local_y == 1) ? -1 : (local_y == blockDim.y) ? 1 : 0;
        offset_z = (local_z == 1) ? -1 : (local_z == blockDim.z) ? 1 : 0;
        if (offset_x)   local_g[INDEX(local_x+offset_x, local_y, local_z, ldxx, ldyy)] = in[INDEX(x+offset_x, y, z, ldx, ldy)];
        if (offset_y)   local_g[INDEX(local_x, local_y+offset_y, local_z, ldxx, ldyy)] = in[INDEX(x, y+offset_y, z, ldx, ldy)];
        if (offset_z)   local_g[INDEX(local_x, local_y, local_z+offset_z, ldxx, ldyy)] = in[INDEX(x, y, z+offset_z, ldx, ldy)];
        
        __syncthreads();
        
        out[INDEX(x, y, z, ldx, ldy)] \
            = 
            ALPHA_ZZZ * local_g[INDEX(local_x, local_y, local_z, ldxx, ldyy)] \
            + ALPHA_NZZ * local_g[INDEX(local_x-1, local_y, local_z, ldxx, ldyy)] \
            + ALPHA_PZZ * local_g[INDEX(local_x+1, local_y, local_z, ldxx, ldyy)] \
            + ALPHA_ZNZ * local_g[INDEX(local_x, local_y-1, local_z, ldxx, ldyy)] \
            + ALPHA_ZPZ * local_g[INDEX(local_x, local_y+1, local_z, ldxx, ldyy)] \
            + ALPHA_ZZN * local_g[INDEX(local_x, local_y, local_z-1, ldxx, ldyy)] \
            + ALPHA_ZZP * local_g[INDEX(local_x, local_y, local_z+1, ldxx, ldyy)];
    }
}


__device__ __forceinline__
void readBlockAndHalo_27(cptr_t  in, const int k, ptr_t  shm, const int shm_ptr, \
                    int & global_ptr, const int ldx, const int ldy, \
                    const int tx,       const int ty,       const int ldxx) {
    __syncthreads();// 因为要对shm进行写操作，防止还有线程没做完shm2regs

    shm[shm_ptr] = in[global_ptr];// +1 for halo offset

    // 要知道 不太可能有一个线程同时会拷贝八个方向的halo！减少分支！
    int offset_x, offset_y;
    offset_x = (tx == 0) ? -1 : (tx == blockDim.x-1) ? 1 : 0;
    offset_y = (ty == 0) ? -1 : (ty == blockDim.y-1) ? 1 : 0;
    if (offset_x)               shm[shm_ptr + offset_x                ] = in[global_ptr + offset_x               ];
    if (offset_y)               shm[shm_ptr + offset_y*ldxx           ] = in[global_ptr + offset_y*ldx           ];
    if (offset_x && offset_y)   shm[shm_ptr + offset_x + offset_y*ldxx] = in[global_ptr + offset_x + offset_y*ldx];

    global_ptr += ldx*ldy;
}

__device__ __forceinline__
void readBlockAndHalo_27_32x6(cptr_t  in, const int k, ptr_t  shm, const int ldx, const int ldy) {
    __syncthreads();// 因为要对shm进行写操作，防止还有线程没做完shm2regs
    // input thread block size is 32x6
    // convert thread x/y ids to linead index 0...191
    int ti = threadIdx.y * blockDim.x + threadIdx.x;// block内的局部序号
    // remap the linear index onto a thread block of size 48x4
    int tx  = ti % 48;
    int ty1 = ti / 48;
    int ty2 = ty1 + 4;
    // calculate input array indices
    int ix  = blockIdx.x*blockDim.x + tx - 8;// 8 words offset due to halo overhead
    int iy1 = blockIdx.y*blockDim.y + ty1 - 1;
    int iy2 = blockIdx.y*blockDim.y + ty2 - 1;
    // data and halos is read as two read instructions of thread blocks of size 48x4
    shm[tx + ty1*48] = in[ix + iy1*ldx + k*ldx*ldy];
    shm[tx + ty2*48] = in[ix + iy2*ldx + k*ldx*ldy];
}

__global__ void stencil_27_2d_kernel_1step(cptr_t  in, ptr_t  out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    extern __shared__ data_t _shm[];
    ptr_t shm = _shm;

    // assert(halo_x==1 && halo_y==1);
    const int ldx = nx + 2*halo_x;// 全局数组的宽度
    const int ldy = ny + 2*halo_y;
    const int ldxx = blockDim.x + 2*halo_x;// 局部数组（共享内存的宽度）

    const int tx = threadIdx.x;// 线程负责的数据在局部数组中的xy坐标（不含halo offset）
    const int ty = threadIdx.y;
    const int global_x = blockIdx.x * blockDim.x + tx;// 线程负责的数据在全局数组中的xy坐标（不含halo offset）
    const int global_y = blockIdx.y * blockDim.y + ty;
    int global_ptr = INDEX(global_x+1, global_y+1, 0, ldx, ldy);

    if (global_x < nx && global_y < ny) {
        const int shm_ptr = tx+halo_x + (ty+halo_y)*ldxx;// 线程负责的数据在局部数组中的索引（含halo offset）
        out += global_ptr;//global_x+halo_x + (global_y+halo_y)*ldx;// adjust per-thread output pointer

        data_t r_NN, r_NZ, r_NP;// values in neighboring grid points are explicitly
        data_t r_ZN, r_ZZ, r_ZP;// read from shm to registers before computations
        data_t r_PN, r_PZ, r_PP;

        data_t t1, t2;// intermediate stencil results

        readBlockAndHalo_27(in, 0, shm, shm_ptr, global_ptr, ldx, ldy, tx, ty, ldxx);// read block of data from 0-th XY-plane
        // readBlockAndHalo_27_32x6(in, 0, shm, ldx, ldy);

        shm2regs_27(shm, shm_ptr)

        readBlockAndHalo_27(in, 1, shm, shm_ptr, global_ptr, ldx, ldy, tx, ty, ldxx);
        // readBlockAndHalo_27_32x6(in, 1, shm, ldx, ldy);

        t1 = computeStencil_planeN_27();

        shm2regs_27(shm, shm_ptr)

        readBlockAndHalo_27(in, 2, shm, shm_ptr, global_ptr, ldx, ldy, tx, ty, ldxx);
        // readBlockAndHalo_27_32x6(in, 2, shm, ldx, ldy);

        t2 = computeStencil_planeN_27();
        t1 += computeStencil_planeZ_27();

        for (int k = 1; k <= nz-halo_z; k++) {
            shm2regs_27(shm, shm_ptr)
            
            readBlockAndHalo_27(in, k+2, shm, shm_ptr, global_ptr, ldx, ldy, tx, ty, ldxx);// 从global_memory读入第k+2层的
            // readBlockAndHalo_27_32x6(in, k+2, shm, ldx, ldy);

            out += ldx*ldy;
            out[0] = t1 + computeStencil_planeP_27();

            t1 = t2 + computeStencil_planeZ_27();
            t2 = computeStencil_planeN_27();
        }
        shm2regs_27(shm, shm_ptr)
        out += ldx*ldy;
        out[0] = t1 + computeStencil_planeP_27();
    } 
    // else {
    //     printf(" tx: %d, ty: %d, gx: %d, gy: %d, global_ptr: %d\n", tx, ty, global_x, global_y, global_ptr);
    // }
}


// 将num除以den向上取整
inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    int TILE_X = 1, TILE_Y = 1, TILE_Z = -1;
    dim3 grid_size, block_size;

    switch (nx)
    {
    case 256:
        TILE_X = 32; TILE_Y = 2  ; TILE_Z = 2;// z 不能等于1，否则计算出错！
        break;
    case 384:
        TILE_X = 32; TILE_Y = 2  ; TILE_Z = 8;
        break;
    case 512:
        TILE_X = 32; TILE_Y = 2  ; TILE_Z = 16;
        break;
    default:
        printf(" Not optimal partition! TX: %d, TY: %d, TZ: %d\n", TILE_X, TILE_Y, TILE_Z);
        break;
    }
    if (TILE_Z == -1) {// 2D 
        grid_size.x = ceiling(nx, TILE_X);
        grid_size.y = ceiling(ny, TILE_Y);
        block_size.x = TILE_X;
        block_size.y = TILE_Y;
    } else {
        assert(TILE_Z > 0);
        grid_size.x = ceiling(nx, TILE_X);
        grid_size.y = ceiling(ny, TILE_Y);
        grid_size.z = ceiling(nz, TILE_Z);
        block_size.x = TILE_X;
        block_size.y = TILE_Y;
        block_size.z = TILE_Z;
    }

    printf("grid: %d %d %d\n", grid_size.x, grid_size.y, grid_size.z);
    printf("blok: %d %d %d\n", block_size.x, block_size.y, block_size.z);

    cudaFuncSetCacheConfig(stencil_7_naive_kernel_1step, cudaFuncCachePreferShared); printf("Prefer Shared\n");

    for(int t = 0; t < nt; ++t) {
        stencil_7_naive_kernel_1step<<<grid_size, block_size, sizeof(data_t)*(TILE_X+2)*(TILE_Y+2)*(TILE_Z+2)>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    int TILE_X = 1, TILE_Y = 1, TILE_Z = -1;
    dim3 grid_size, block_size;

    switch (nx)
    {
    case 256:
        TILE_X = 32; TILE_Y = 28;
        break;
    case 384:
        TILE_X = 128; TILE_Y = 8 ;
        break;
    case 512:
        TILE_X = 104; TILE_Y = 8 ;
        break;
    default:
        printf(" Not optimal partition! TX: %d, TY: %d, TZ: %d\n", TILE_X, TILE_Y, TILE_Z);
        break;
    }
    // TILE_X = 32; TILE_Y = 6;
    // assert(nx % TILE_X == 0);
    // assert(ny % TILE_Y == 0);

    grid_size.x = ceiling(nx, TILE_X);
    grid_size.y = ceiling(ny, TILE_Y);
    block_size.x = TILE_X;
    block_size.y = TILE_Y;
    

    printf("grid: %d %d %d\n", grid_size.x, grid_size.y, grid_size.z);
    printf("blok: %d %d %d\n", block_size.x, block_size.y, block_size.z);

    cudaFuncSetCacheConfig(stencil_27_2d_kernel_1step, cudaFuncCachePreferShared); printf("Prefer Shared\n");

    for(int t = 0; t < nt; ++t) {
        stencil_27_2d_kernel_1step<<<grid_size, block_size, sizeof(data_t)*(TILE_Y+2)*(TILE_X+2)>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
        // stencil_27_2d_kernel_1step<<<grid_size, block_size, sizeof(data_t)*(TILE_Y+2)*(TILE_X+2*8)>>>(\
        //     buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
        //         grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}