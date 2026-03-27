# stencil_GPU
Stencil computation on NVidia GPU (Tesla V100)

For detail implementation, please refer to my blog: https://blog.csdn.net/weixin_43614211/article/details/122110207

# 以下为《Stencil计算：NVidia GPU》博客的内容
（由于上述的CSDN需要付费才能看到完整博客内容，违背本人技术分享的初衷，此处附载博客全文）

机器：NVidia GPU Tesla V100单卡

7点stencil的naive版本代码如下：
```c
ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, BLOCK_X), ceiling(ny, BLOCK_Y), ceiling(nz, BLOCK_Z));
    dim3 block_size (BLOCK_X, BLOCK_Y, BLOCK_Z);
    for(int t = 0; t < nt; ++t) {
        stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}

__global__ void stencil_7_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;// 线程负责的数据在全局的坐标（但注意不含halo的offset！）
    int ty = threadIdx.y  + blockDim.y * blockIdx.y;
    int tz = threadIdx.z  + blockDim.z * blockIdx.z;
    int ldxx = BLOCK_X+2;// local_g的宽度
    int ldyy = BLOCK_Y+2;

    __shared__ double local_g[(BLOCK_X+2)*(BLOCK_Y+2)*(BLOCK_Z+2)];
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
        offset_x = (local_x == 1) ? -1 : (local_x == BLOCK_X) ? 1 : 0;
        offset_y = (local_y == 1) ? -1 : (local_y == BLOCK_Y) ? 1 : 0;
        offset_z = (local_z == 1) ? -1 : (local_z == BLOCK_Z) ? 1 : 0;
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
```
类似地，27点的naive代码同理。

对于naïve版本的代码，不妨先“无脑”地加上`-O3 -arch=compute_70 -code=compute_70`等编译选项。但并不能获得任何的性能提升。

查阅文献，感觉GPU优化的套路较为固定，大部分都是利用shared_memory做访存优化，并适当利用寄存器做额外的计算（e.g. 一个线程负责多个输出元素的计算）。部分文献还有利用GPU访问global_memory时的聚合访问（coalesced transaction）特性，来特别设计kernel以避免halo区附近线程的指令流divergence。

在Anamaria Vizitiu写的[Optimized Three-Dimensional Stencil Computation on Fermi and Kepler GPUs](https://ieeexplore.ieee.org/document/7040968)，和Paulius Micikevicius写的[3D Finite Difference Computation on GPUs using CUDA](https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf)（左下图代码），和Marcin Krotkiewski写的[Efficient 3D stencil computations using CUDA](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.3680&rep=rep1&type=pdf)（右下图伪代码）中，都提到用二维划分做Tiling是最好的。Tiling的示意图同CPU版本，利用shared_memory，每次将一个z平面的数据读入到共享内存中，然后做相应计算，计算完成后将共享内存的数据转移到各个线程自己的寄存器上，然后再读入下一个z平面的数据，循环往复……

<table>
    <tr>
        <td ><center><img src="https://i-blog.csdnimg.cn/blog_migrate/e4e437ff6081fedf93629fe7c469155b.png" ></center></td>
        <td ><center><img src="https://i-blog.csdnimg.cn/blog_migrate/fd224db9551b070516097e25af1309b3.png"  ></center></td>
    </tr>
</table>

这两种实现的区别在于，左边的代码（也是常见的stencil计算方法）是**从输出平面的角度来表述的**：对于每个输出点，从全局内存中读取所需的数据，计算模板，并存储结果。在GPU上，计算第k个输出平面的stencil是通过读取第k-1，k，k+1三个输入平面到线程块的共享内存中实现的。在下一个第k+1个输出平面的计算中，可以重用第k和k+1个输入平面的数据。

相比之下，右边的算法是**从输入平面的角度来表述的**：线程块只从全局内存中读取一个XY数据块到共享内存中（包括halo区）。这个来自第k个输入平面的分块被用作三个输出平面（k-1，k，k+1）的stencil计算的一部分。每单个平面的计算在computeStencil_plane0... 2子函数中。 将获得的部分stencil结果累积到寄存器t1和t2中。 一旦三个相邻平面的部分结果累积起来，结果就存储在输出数组中(34行)。

右边的方法的优点是，每个网格点只使用9条load指令（shm2regs函数中）就可以将数据值**从共享内存加载到寄存器中**。而在共享内存中存储三个输入平面的通常方法（左边的做法，共享内存存一个输入平面，寄存器存两个输入平面）中，值要么在计算之前直接从共享内存加载到寄存器中（每个网格点27条load指令，假定左边的代码也是27点stencil），要么算术运算必须使用共享内存中的操作数。而**使用共享内存操作数的算术运算要比使用寄存器的更慢**。

所以按照Marcin Krotkiewski写的[Efficient 3D stencil computations using CUDA](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.3680&rep=rep1&type=pdf)，实现了如下所示的kernel，以27点为例。

```c
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
}
```

同时利用条件跳转语句，减少了原文中`readBlockAndHalo`函数中的一些if-else分支,以避免一个warp内的指令流divergence带来的性能损失。

```c
#define computeStencil_planeN_27()    ALPHA_NNN * r_NN + ALPHA_NZN * r_NZ + ALPHA_NPN * r_NP \
                                    + ALPHA_ZNN * r_ZN + ALPHA_ZZN * r_ZZ + ALPHA_ZPN * r_ZP \
                                    + ALPHA_PNN * r_PN + ALPHA_PZN * r_PZ + ALPHA_PPN * r_PP
#define computeStencil_planeZ_27()    ALPHA_NNZ * r_NN + ALPHA_NZZ * r_NZ + ALPHA_NPZ * r_NP \
                                    + ALPHA_ZNZ * r_ZN + ALPHA_ZZZ * r_ZZ + ALPHA_ZPZ * r_ZP \
                                    + ALPHA_PNZ * r_PN + ALPHA_PZZ * r_PZ + ALPHA_PPZ * r_PP
#define computeStencil_planeP_27()    ALPHA_NNP * r_NN + ALPHA_NZP * r_NZ + ALPHA_NPP * r_NP \
                                    + ALPHA_ZNP * r_ZN + ALPHA_ZZP * r_ZZ + ALPHA_ZPP * r_ZP \
                                    + ALPHA_PNP * r_PN + ALPHA_PZP * r_PZ + ALPHA_PPP * r_PP
#define shm2regs_27(shm, shm_ptr)  __syncthreads(); \
                    r_NN = shm[shm_ptr - 1 - ldxx]; r_NZ = shm[shm_ptr - 1       ]; r_NP = shm[shm_ptr - 1 + ldxx]; \
                    r_ZN = shm[shm_ptr     - ldxx]; r_ZZ = shm[shm_ptr           ]; r_ZP = shm[shm_ptr     + ldxx]; \
                    r_PN = shm[shm_ptr + 1 - ldxx]; r_PZ = shm[shm_ptr + 1       ]; r_PP = shm[shm_ptr + 1 + ldxx];

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
```
按此实现的27点stencil能获得1900~2000 GFlop/s的性能。此部分代码见stencil-optimized.cu文件。
| 算例规模 | 最佳Tiling大小 | 性能 (GFlop/s) |
|--|--|--|
| 256 | X=32, Y=28 | 1986.18 |
| 384 | X=128, Y=8 | 2006.52 |
| 512 | X=104, Y=8 | 1904.46 |

如法炮制7点stencil，但发现性能不增反降。而且仅仅通过调整naïve代码中的`BLOCK_X`，`BLOCK_Y`，`BLOCK_Z`的分块参数，就能获得不小的性能提升。例如：
| 算例规模 | 分块 | 7点 (GFlop/s) | 27点 (GFlop/s) |
|--|--|--|--|
| 256 | X=32, Y=8, Z=4 | 482.00 | 1622.98 |
| 384 | X=32, Y=8, Z=4 | 481.50 | 1607.73 |
| 512 | X=32, Y=8, Z=4 | 491.10 | 1633.75 |

考虑这个结果与文献分析之间的差异的可能原因。所阅读到的认为2D tiling比3D blocking更有效的文献基本都在2014年以前出版，给出的理由基本为每个thread block内的shared_memory容量有限（彼时常为16 KB，能容纳$12^3$或$45^2$个双精度数），不足以容纳充分大的三维分块，因此用2D tiling更节省shared_memory。而目前所用的Tesla V100 GPU的每个block已经可以最多使用到96 KB（能容纳$23^3$或$110^2$个双精度数）的shared_memory，因此使三维线程块的高效使用成为可能。

在Marcin Krotkiewski写的[Efficient 3D stencil computations using CUDA](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.3680&rep=rep1&type=pdf)一文中，还对单精度的计算做了访问global_memory的优化，如下图示意，在将数据从global_memory读入shared_memory时，将$32\times6$的线程块重组成$48\times4$的维度，以利用显存的聚合访问（coalesced transaction）特性，使得每个线程都“有活可干”，且分到的“活”任务量相等，不出现靠近halo区的线程要多加载部分数据的if-else分支情况。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a3001f87ce3567660ee5016a4ac351c7.png#pic_center)

但这种做法具有一定取巧性，在单精度、聚合访存长度为32 Bytes的情况下，刚好能如此重组。而在本次作业的机器中，在双精度、聚合访存长度为32 Bytes的情况下，则需要将$64\times6$的线程块重组为$96\times4$，并且每个线程要做两次读操作，负责两个输出位置的计算。代码实现起来将较为复杂，因此我没有实现这种优化。但这种分析memory footprint的访存优化方法，确实是有意思和启发的。

最终，GPU版本的stencil计算性能总结如下。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/66dc2275f69a62447f346bb4ccdabce7.png#pic_center)

