#include <torch/extension.h>

#define COMPUTE_COLOR_POINTERS() \
    scalar_t*  img_color_ptr = img_ptr + color * img_color_stride;\
    const scalar_t*  grad_color_ptr = grad_ptr + color * grad_color_stride;

#define COMPUTE_ROW_POINTERS() \
    scalar_t* img_row_ptr = img_color_ptr + y * img_row_stride;\
    scalar_t* img_row_ptr_0 = img_row_ptr;\
    scalar_t* img_row_ptr_1 = img_color_ptr + (y+1) * img_row_stride;\
    scalar_t* img_row_ptr_2 = img_color_ptr + (y+2) * img_row_stride;\
    scalar_t* img_row_ptr_3 = img_color_ptr + (y+3) * img_row_stride;\
    scalar_t* img_row_ptr_4 = img_color_ptr + (y+4) * img_row_stride;\
    scalar_t* img_row_ptr_5 = img_color_ptr + (y+5) * img_row_stride;\
    scalar_t* img_row_ptr_6 = img_color_ptr + (y+6) * img_row_stride;\
    scalar_t* img_row_ptr_7 = img_color_ptr + (y+7) * img_row_stride;\
    scalar_t* above_img_row_ptr = img_color_ptr + (y-1) * img_row_stride;\
    scalar_t* below_img_row_ptr = img_row_ptr_1;\
    const scalar_t* grad_row_ptr = grad_color_ptr + y * grad_row_stride;\
    const scalar_t* grad_row_ptr_0 = grad_row_ptr;\
    const scalar_t* grad_row_ptr_1 = grad_color_ptr + (y+1) * grad_row_stride;\
    const scalar_t* grad_row_ptr_2 = grad_color_ptr + (y+2) * grad_row_stride;\
    const scalar_t* grad_row_ptr_3 = grad_color_ptr + (y+3) * grad_row_stride;\
    const scalar_t* grad_row_ptr_4 = grad_color_ptr + (y+4) * grad_row_stride;\
    const scalar_t* grad_row_ptr_5 = grad_color_ptr + (y+5) * grad_row_stride;\
    const scalar_t* grad_row_ptr_6 = grad_color_ptr + (y+6) * grad_row_stride;\
    const scalar_t* grad_row_ptr_7 = grad_color_ptr + (y+7) * grad_row_stride;\
    const scalar_t* above_grad_row_ptr = grad_color_ptr + (y-1) * grad_row_stride;


void step_cpu(int step, torch::Tensor img, torch::Tensor grad){

    // Pre-Compute some commonly used variables, for a little extra speed (works, measured)
    const int num_colors = img.size(0);
    const int height = img.size(1);
    const int width = img.size(2);
    const int img_color_stride = img.stride(0);
    const int img_row_stride = img.stride(1);
    const int grad_color_stride = grad.stride(0);
    const int grad_row_stride = grad.stride(1);

    // Automatically determine the data type of our CPU kernels
    AT_DISPATCH_FLOATING_TYPES(img.type(), "step_cpu", ([&] {
        // Get raw pointers
        scalar_t* img_ptr = img.data<scalar_t>();
        const scalar_t* grad_ptr = grad.data<scalar_t>();

        switch(step%4){
            case 0: // Left to Right
                for(int color = 0; color < num_colors; color++){
                    COMPUTE_COLOR_POINTERS();
                    int y = 1;
                    // Unrolling improved speed x4, as it enabled vectorization
                    for(; y < height-1-7; y+=8){
                        COMPUTE_ROW_POINTERS();
                        for(int x = 1; x < width-1; x++){
                            img_row_ptr_0[x] = (img_row_ptr_0[x] + img_row_ptr_0[x - 1] + grad_row_ptr_0[x - 1]) / 2;
                            img_row_ptr_1[x] = (img_row_ptr_1[x] + img_row_ptr_1[x - 1] + grad_row_ptr_1[x - 1]) / 2;
                            img_row_ptr_2[x] = (img_row_ptr_2[x] + img_row_ptr_2[x - 1] + grad_row_ptr_2[x - 1]) / 2;
                            img_row_ptr_3[x] = (img_row_ptr_3[x] + img_row_ptr_3[x - 1] + grad_row_ptr_3[x - 1]) / 2;
                            img_row_ptr_4[x] = (img_row_ptr_4[x] + img_row_ptr_4[x - 1] + grad_row_ptr_4[x - 1]) / 2;
                            img_row_ptr_5[x] = (img_row_ptr_5[x] + img_row_ptr_5[x - 1] + grad_row_ptr_5[x - 1]) / 2;
                            img_row_ptr_6[x] = (img_row_ptr_6[x] + img_row_ptr_6[x - 1] + grad_row_ptr_6[x - 1]) / 2;
                            img_row_ptr_7[x] = (img_row_ptr_7[x] + img_row_ptr_7[x - 1] + grad_row_ptr_7[x - 1]) / 2;
                        }
                    }
                    for(; y < height-1; y++){
                        COMPUTE_ROW_POINTERS();
                        for(int x = 1; x < width-1; x++){
                            img_row_ptr_0[x] = (img_row_ptr_0[x] + img_row_ptr_0[x - 1] + grad_row_ptr_0[x - 1]) / 2;
                        }
                    }
                }
                break;

            case 1: // Top to Bottom
                // No unrolling here, as rows depend on each other and cannot be parallelized.
                // Line-wise unrolling happens automatically, for row-wise hat to be done manually as the compiler
                // doesn't vectorize through nested loops.
                for(int color = 0; color < num_colors; color++){
                    COMPUTE_COLOR_POINTERS();
                    for(int y = 1; y < height-1; y++){
                        COMPUTE_ROW_POINTERS();
                        for(int x = 1; x < width-1; x++){
                            img_row_ptr[x] = (img_row_ptr[x] + above_img_row_ptr[x] + above_grad_row_ptr[x]) / 2;
                        }
                    }
                }
                break;

            case 2:// Right to Left
                for(int color = 0; color < num_colors; color++){
                    COMPUTE_COLOR_POINTERS();
                    int y = 1;
                    // Unrolling improved speed x4, as it enabled vectorization
                    for(; y < height-1-7; y+=8){
                        COMPUTE_ROW_POINTERS();
                        for(int x = width-2; x > 0; x--){
                            img_row_ptr_0[x] = (img_row_ptr_0[x] + img_row_ptr_0[x + 1] - grad_row_ptr_0[x]) / 2;
                            img_row_ptr_1[x] = (img_row_ptr_1[x] + img_row_ptr_1[x + 1] - grad_row_ptr_1[x]) / 2;
                            img_row_ptr_2[x] = (img_row_ptr_2[x] + img_row_ptr_2[x + 1] - grad_row_ptr_2[x]) / 2;
                            img_row_ptr_3[x] = (img_row_ptr_3[x] + img_row_ptr_3[x + 1] - grad_row_ptr_3[x]) / 2;
                            img_row_ptr_4[x] = (img_row_ptr_4[x] + img_row_ptr_4[x + 1] - grad_row_ptr_4[x]) / 2;
                            img_row_ptr_5[x] = (img_row_ptr_5[x] + img_row_ptr_5[x + 1] - grad_row_ptr_5[x]) / 2;
                            img_row_ptr_6[x] = (img_row_ptr_6[x] + img_row_ptr_6[x + 1] - grad_row_ptr_6[x]) / 2;
                            img_row_ptr_7[x] = (img_row_ptr_7[x] + img_row_ptr_7[x + 1] - grad_row_ptr_7[x]) / 2;
                        }
                    }
                    for(; y < height-1; y++){
                        COMPUTE_ROW_POINTERS();
                        for(int x = width-2; x > 0; x--){
                            img_row_ptr_0[x] = (img_row_ptr_0[x] + img_row_ptr_0[x+1] - grad_row_ptr_0[x]) / 2;
                        }
                    }
                }
                break;

            default: // Bottom to Top
                for(int color = num_colors-1; color >= 0; color--){
                    COMPUTE_COLOR_POINTERS();
                    for(int y = height-2; y > 0; y--){
                        COMPUTE_ROW_POINTERS();
                        for(int x = width-2; x > 0; x--){
                            img_row_ptr[x] = (img_row_ptr[x] + below_img_row_ptr[x] - grad_row_ptr[x]) / 2;
                        }
                    }
                }
                break;
        }
    }));
}

