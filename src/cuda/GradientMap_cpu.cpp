#include <torch/extension.h>

// A macro to remove some boilerplate code.
#define RECONSTRUCTION_FUNCTION(name, code) \
void name ## _cpu(\
    torch::TensorAccessor<float,3> img,\
    const torch::TensorAccessor<float,3>  grad){\
    for(int col = 0; col < img.size(0); col++){code}}


RECONSTRUCTION_FUNCTION(lr_kernel,{
    for(int y = 1; y < img.size(1)-1; y++){
        for(int x = 1; x < img.size(2)-1; x++){
            img[col][y][x] = (img[col][y][x] + img[col][y][x - 1] + grad[col][y][x - 1]) / 2;
        }
    }
})

RECONSTRUCTION_FUNCTION(rl_kernel,{
    for(int y = 1; y < img.size(1)-1; y++){
        for(int x = img.size(2)-2; x > 0; x--){
            img[col][y][x] = (img[col][y][x] + img[col][y][x + 1] - grad[col][y][x]) / 2;
        }
    }
})

RECONSTRUCTION_FUNCTION(tb_kernel,{
    for(int y = 1; y < img.size(1)-1; y++){
        for(int x = 1; x < img.size(2)-1; x++){
            img[col][y][x] = (img[col][y][x] + img[col][y-1][x] + grad[col][y-1][x]) / 2;
        }
    }
})

RECONSTRUCTION_FUNCTION(bt_kernel,{
    for(int y = img.size(1)-2; y > 0; y--){
        for(int x = 1; x < img.size(2)-1; x++){
            img[col][y][x] = (img[col][y][x] + img[col][y+1][x] - grad[col][y][x]) / 2;
        }
    }
})


void step_cpu(int step, torch::Tensor img, torch::Tensor grad){

    // Pack into accessors
    auto img_accessor = img.accessor<float,3>();
    auto grad_accessor = grad.accessor<float,3>();

    switch(step%4){
        case 0:
            lr_kernel_cpu(img_accessor, grad_accessor);
            break;
        case 1:
            tb_kernel_cpu(img_accessor, grad_accessor);
            break;
        case 2:
            rl_kernel_cpu(img_accessor, grad_accessor);
            break;
        default:
            bt_kernel_cpu(img_accessor, grad_accessor);
            break;
    }
}
