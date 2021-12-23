#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "common.h"

extern const char* version_name;

int parse_args(dist_grid_info_t *info, int *nsteps, int* stencil_type, int argc, char **argv);
int read_file(ptr_t grid, const char *file_name, const dist_grid_info_t* info, ptr_t buffer);
ptr_t memory_alloc_device(const dist_grid_info_t *info);
ptr_t memory_alloc_host(const dist_grid_info_t *info);

typedef struct {
    data_t norm_1, norm_2, norm_inf;
} check_result_t;

extern check_result_t check_answer(cptr_t ans0, cptr_t ans1, const dist_grid_info_t *info);

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != 0, err_code)
#define CHECK_NULL(ret, err_code) CHECK(ret == NULL, err_code)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, (exit(1), 1))
#define ABORT_IF_NULL(ret) CHECK(ret == NULL, (exit(1), 1))

int main(int argc, char **argv) {
    struct timeval start, end;
    double run_time, pre_time, gflops;
    int nt, type;
    dist_grid_info_t info;
    ptr_t a0, a1, ans0, ans1, host_buffer;

    ABORT_IF_ERROR(parse_args(&info, &nt, &type, argc, argv))

    gettimeofday(&start, NULL);
    create_dist_grid(&info, type);
    gettimeofday(&end, NULL);
    pre_time = (double)(end.tv_sec - start.tv_sec) + 1e-6 * (double)(end.tv_usec - start.tv_usec);

    a0 = memory_alloc_device(&info);
    a1 = memory_alloc_device(&info);
    host_buffer = memory_alloc_host(&info);
    ABORT_IF_NULL(a0)
    ABORT_IF_NULL(a1)
    ABORT_IF_NULL(host_buffer)
    ABORT_IF_ERROR(read_file(a0, argv[6], &info, host_buffer));// read initial data

    gettimeofday(&start, NULL);
    if (type == 7) {
        ans0 = stencil_7(a0, a1, &info, nt);
    } else {
        ans0 = stencil_27(a0, a1, &info, nt);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    run_time = (double)(end.tv_sec - start.tv_sec) + 1e-6 * (double)(end.tv_usec - start.tv_usec);

    if(argc == 8) {// check answer
        check_result_t result;
        ans1 = (ans0 == a0) ? a1 : a0;
        ABORT_IF_ERROR(read_file(ans1, argv[7], &info, host_buffer))// read reference data
        result = check_answer(ans0, ans1, &info);
        printf("errors:\n    1-norm = %.16lf\n    2-norm = %.16lf\n  inf-norm = %.16lf\n", \
                result.norm_1, result.norm_2, result.norm_inf);
        if(result.norm_inf > 1e-9) {
            printf("Significant numeric error.\n");
        }
    } else { 
        printf("Result unchecked.\n");
    }

    gflops = 1e-9 * info.global_size_x * info.global_size_y * info.global_size_z\
             * nt * (type * 2 - 1) / run_time;

    printf("%d-point stencil - %s:\nSize (%d x %d x %d), Timestep %d\n",type, version_name, info.global_size_x, info.global_size_y, info.global_size_z, nt);
    printf("Preprocessing time %lfs\n", pre_time);
    printf("Computation time %lfs, Performance %lfGflop/s\n", run_time, gflops);
    cudaFreeHost(host_buffer);
    cudaFree(a0);
    cudaFree(a1);
    return 0;
}

void print_help(const char *argv0) {
    printf("USAGE: %s <stencil-type> <nx> <ny> <nz> <nt> <input-file> [<answer-file>]\n  where <stencil-type> is 7 or 27\n", argv0);
}

int parse_args(dist_grid_info_t *info, int *nsteps, int* stencil_type, int argc, char **argv) {
    int nx, ny, nz, nt, type;
    if(argc < 7) {
        print_help(argv[0]);
        return 1;
    } 
    type = atoi(argv[1]);
    nx = atoi(argv[2]);
    ny = atoi(argv[3]);
    nz = atoi(argv[4]);
    nt = atoi(argv[5]);
    if(nx <= 0 || ny <= 0 || nz <= 0 || nt <= 0 || (type != 7 && type != 27)) {
        print_help(argv[0]);
        return 1;
    }
    info->global_size_x = nx;
    info->global_size_y = ny;
    info->global_size_z = nz;
    *nsteps = nt;
    *stencil_type = type;
    return 0;
}

// 将file_name内的数据拷贝到host端的buffer，同时拷贝到device端的grid(放入grid的核心区，绕开halo)
int read_file(ptr_t grid, const char *file_name, const dist_grid_info_t* info, ptr_t buffer) {
    int nx = info->global_size_x;
    int ny = info->global_size_y;
    int nz = info->global_size_z;
    int halo_x = info->halo_size_x;
    int halo_y = info->halo_size_y;
    int halo_z = info->halo_size_z;
    int ldx = nx + 2 * halo_x;
    int ldy = ny + 2 * halo_y;
    int count;
    FILE *file = fopen(file_name, "rb");
    CHECK_NULL(file, cudaErrorOperatingSystem);
    count = fread(buffer, sizeof(data_t), nx*ny*nz, file);// 读入到host端的buffer
    fclose(file);
    if(count != nx*ny*nz) {
        return 1;
    }
    for(int z = 0; z < nz; ++z) {
        for(int y = 0; y < ny; ++y) {// 只将核心区域的数据拷贝到device端？
            if(cudaMemcpy(grid + INDEX(halo_x, y+halo_y, z+halo_z, ldx, ldy), \
                       buffer + INDEX(0, y, z, nx, ny), \
                       sizeof(data_t) * nx, cudaMemcpyHostToDevice) != cudaSuccess) {
                return 1;
            }
        }
    }
    return 0;
}

int set_zero(ptr_t buff, int size) {
    if(cudaMemset(buff, 0, sizeof(data_t) * size) != cudaSuccess) {
        return 1;
    }
    return 0;
}

ptr_t memory_alloc_device(const dist_grid_info_t *info) {
    ptr_t tmp;
    int x_end = info->global_size_x + info->halo_size_x;
    int y_start = info->halo_size_y, y_end = info->global_size_y + info->halo_size_y;
    int z_start = info->halo_size_z, z_end = info->global_size_z + info->halo_size_z;
    int nx = info->global_size_x + 2 * info->halo_size_x;
    int ny = info->global_size_y + 2 * info->halo_size_y;
    int nz = info->global_size_z + 2 * info->halo_size_z;
    int ret = cudaMalloc(&tmp, sizeof(data_t) * nx * ny * nz);
    CHECK_ERROR(ret, NULL);
    ret = set_zero(tmp+INDEX(0,0,0,nx,ny), nx*ny*info->halo_size_z);
    for(int z = z_start; z < z_end; ++z) {
        ret |= set_zero(tmp+INDEX(0,0,z,nx,ny), nx*info->halo_size_y);
        for(int y = y_start; y < y_end; ++y) {
            ret |= set_zero(tmp+INDEX(0,y,z,nx,ny), info->halo_size_x);
            ret |= set_zero(tmp+INDEX(x_end,y,z,nx,ny), info->halo_size_x);
        }
        ret |= set_zero(tmp+INDEX(0,y_end,z,nx,ny), nx*info->halo_size_y);
    }
    ret |= set_zero(tmp+INDEX(0,0,z_end,nx,ny), nx*ny*info->halo_size_z);
    if(ret != 0) {
        cudaFree(tmp);
        return NULL;
    }
    return tmp;
}

ptr_t memory_alloc_host(const dist_grid_info_t *info) {// 为什么device端分配的是global_size_而不带halo？
    ptr_t tmp;
    int nx = info->global_size_x;
    int ny = info->global_size_y;
    int nz = info->global_size_z;
    int ret = cudaMallocHost(&tmp, sizeof(data_t) * nx * ny * nz);
    CHECK_ERROR(ret, NULL);
    return tmp;
}

