#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


// A macro to remove all the messy boilerplate code.
// Defines the function name and sets up the convenience variables 'index', 'stride' and 'col'.
// 'index' and 'stride' is a great design pattern that I will use always from now on, which i found in:
// https://devblogs.nvidia.com/even-easier-introduction-cuda/
// The amazing part is that you can change the block and dimension size without having to adjust the kernel,
// for performance comparison. It will automatically adjust and still compute all the necessary data.
#define RECONSTRUCTION_FUNCTION(name, code)                                                         \
template <typename scalar_t>                                                                        \
__global__ void name ## _cuda(                                                                      \
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> img,                    \
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t>  grad_x)          \
{                                                                                                   \
    int index = blockIdx.x * blockDim.x + threadIdx.x;                                              \
    int stride = blockDim.x * gridDim.x;                                                            \
    int col = blockIdx.y;                                                                           \
    code                                                                                            \
}


RECONSTRUCTION_FUNCTION(lr_kernel,{
    for(int y = index+1; y < img.size(1)-1; y+=stride){
        for(int x = 1; x < img.size(2)-1; x++){
            img[col][y][x] = (img[col][y][x] + img[col][y][x - 1] + grad_x[col][y][x - 1]) / 2;
        }
    }
})

RECONSTRUCTION_FUNCTION(rl_kernel,{
    for(int y = index+1; y < img.size(1)-1; y+=stride){
        for(int x = img.size(2)-2; x > 0; x--){
            img[col][y][x] = (img[col][y][x] + img[col][y][x + 1] - grad_x[col][y][x]) / 2;
        }
    }
})

RECONSTRUCTION_FUNCTION(tb_kernel,{
    for(int x = index+1; x < img.size(2)-1; x+=stride){
        for(int y = 1; y < img.size(1)-1; y++){
            img[col][y][x] = (img[col][y][x] + img[col][y-1][x] + grad_x[col][y-1][x]) / 2;
        }
    }
})

RECONSTRUCTION_FUNCTION(bt_kernel,{
    for(int x = index+1; x < img.size(2)-1; x+=stride){
        for(int y = img.size(1)-2; y > 0; y--){
            img[col][y][x] = (img[col][y][x] + img[col][y+1][x] - grad_x[col][y][x]) / 2;
        }
    }
})


void step_cuda(int step, torch::Tensor img, torch::Tensor grad){

    // Compute wavefront size. This is the number of parallel workers we have.
    // It is perpendicular to our walking direction, eg. the LR kernel has the height 'img' as wavefront.
    const int wavefront_size = (step%2==0)?img.size(1):img.size(2);

    // Compute number of blocks from wavefront
    const int blockSize = 1024;
    const int numBlocks = (wavefront_size + blockSize - 1) / blockSize;

    // Add a second dimension for the colors
    const dim3 numBlocksWithColors(numBlocks,img.size(0));

    // Automatically determine the data type of our CUDA kernels
    AT_DISPATCH_FLOATING_TYPES(img.type(), "step_cuda", ([&] {
        // Create accessors
        auto img_accessor = img.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>();
        auto grad_accessor = grad.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>();

        // The actual step
        switch(step%4){
            case 0:
                lr_kernel_cuda<scalar_t><<<numBlocksWithColors,blockSize>>>(img_accessor, grad_accessor);
                break;
            case 1:
                tb_kernel_cuda<scalar_t><<<numBlocksWithColors,blockSize>>>(img_accessor, grad_accessor);
                break;
            case 2:
                rl_kernel_cuda<scalar_t><<<numBlocksWithColors,blockSize>>>(img_accessor, grad_accessor);
                break;
            default:
                bt_kernel_cuda<scalar_t><<<numBlocksWithColors,blockSize>>>(img_accessor, grad_accessor);
                break;
        }
    }));
}
