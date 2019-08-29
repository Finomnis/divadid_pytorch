#include <torch/extension.h>

// CUDA forward declarations

void step_cuda(int step, torch::Tensor img, torch::Tensor grad);
void step_cpu(int step, torch::Tensor img, torch::Tensor grad);


// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_NOT_CUDA(x) AT_ASSERTM(!x.type().is_cuda(), #x " cannot be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void step(int step, torch::Tensor img, torch::Tensor grad) {
    CHECK_CONTIGUOUS(img);
    CHECK_CONTIGUOUS(grad);

    if (img.type().is_cuda()) {
        CHECK_CUDA(grad);

        step_cuda(step, img, grad);
    } else {
        CHECK_NOT_CUDA(grad);

        step_cpu(step, img, grad);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step, "Reconstruction Step (CUDA)");
}
