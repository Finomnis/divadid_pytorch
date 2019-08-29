#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


#define RECONSTRUCTION_FUNCTION(name, code) \
__global__ void name(\
    torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> img,\
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t>  grad_x){\
    int index = blockIdx.x * blockDim.x + threadIdx.x;\
    int stride = blockDim.x * gridDim.x;\
    int col = blockIdx.y;\
    code}

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


void step_cuda(int step, torch::Tensor img, torch::Tensor grad){
    //const auto img_width = img.size(1)
    // TODO dim3 to parallelize colors
    dim3 num_blocks(4,3);

    switch(step%4){
    case 0:
        lr_kernel<<<num_blocks,1024>>>(
            img.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
            grad.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>());
        break;
    case 1:
        //tb_kernel<<<num_blocks,1024>>>(
        //    img.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        //    grad_x.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>());
        break;
    case 2:
        rl_kernel<<<num_blocks,1024>>>(
            img.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
            grad.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>());
        break;
    default:
        //bt_kernel<<<num_blocks,1024>>>(
        //    img.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
        //    grad_x.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>());
        break;
    }
}
