#include <torch/extension.h>

// THE ACTUAL COMPUTATION LOOPS, AS MACROS, TO MAKE THE MAIN CODE MORE READABLE
#define RECONSTRUCTION_LOOP_TL_TO_BR(code)                                                          \
for(int color = 0; color < num_colors; color++){                                                    \
    scalar_t* img_color_ptr = img_ptr + color * img_color_stride;                                   \
    const scalar_t* grad_color_ptr = grad_ptr + color * grad_color_stride;                          \
    for(int y = 1; y < height-1; y++){                                                              \
        scalar_t* img_row_ptr = img_color_ptr + y * img_row_stride;                                 \
        scalar_t* prev_img_row_ptr = img_color_ptr + (y-1) * img_row_stride;                        \
        const scalar_t* grad_row_ptr = grad_color_ptr + y * grad_row_stride;                        \
        const scalar_t* prev_grad_row_ptr = grad_color_ptr + (y-1) * grad_row_stride;               \
        for(int x = 1; x < width-1; x++){                                                           \
            code                                                                                    \
        }                                                                                           \
    }                                                                                               \
}
#define RECONSTRUCTION_LOOP_BR_TO_TL(code)                                                          \
for(int color = num_colors-1; color >= 0; color--){                                                 \
    scalar_t* img_color_ptr = img_ptr + color * img_color_stride;                                   \
    const scalar_t* grad_color_ptr = grad_ptr + color * grad_color_stride;                          \
    for(int y = height-2; y > 0; y--){                                                              \
        scalar_t* img_row_ptr = img_color_ptr + y * img_row_stride;                                 \
        scalar_t* prev_img_row_ptr = img_color_ptr + (y+1) * img_row_stride;                        \
        const scalar_t* grad_row_ptr = grad_color_ptr + y * grad_row_stride;                        \
        for(int x = width-2; x > 0; x--){                                                           \
            code                                                                                    \
        }                                                                                           \
    }                                                                                               \
}


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
                RECONSTRUCTION_LOOP_TL_TO_BR({
                    // reference: img[col][y][x] = (img[col][y][x] + img[col][y][x - 1] + grad[col][y][x - 1]) / 2;
                    img_row_ptr[x] = (img_row_ptr[x] + img_row_ptr[x - 1] + grad_row_ptr[x - 1]) / 2;
                })
                break;

            case 1: // Top to Bottom
                RECONSTRUCTION_LOOP_TL_TO_BR({
                    // reference: img[col][y][x] = (img[col][y][x] + img[col][y-1][x] + grad[col][y-1][x]) / 2;
                    img_row_ptr[x] = (img_row_ptr[x] + prev_img_row_ptr[x] + prev_grad_row_ptr[x]) / 2;
                })
                break;

            case 2:// Right to Left
                RECONSTRUCTION_LOOP_BR_TO_TL({
                    // reference: img[col][y][x] = (img[col][y][x] + img[col][y][x + 1] - grad[col][y][x]) / 2;
                    img_row_ptr[x] = (img_row_ptr[x] + img_row_ptr[x+1] - grad_row_ptr[x]) / 2;
                })
                break;

            default: // Bottom to Top
                RECONSTRUCTION_LOOP_BR_TO_TL({
                    // reference: img[col][y][x] = (img[col][y][x] + img[col][y+1][x] - grad[col][y][x]) / 2;
                    img_row_ptr[x] = (img_row_ptr[x] + prev_img_row_ptr[x] - grad_row_ptr[x]) / 2;
                })
                break;
        }
    }));
}

