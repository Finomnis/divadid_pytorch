#include <torch/extension.h>

// CPU forward declaration
void step_cpu(int step, torch::Tensor img, torch::Tensor grad);

#ifdef WITH_CUDA
// CUDA forward declarations
void step_cuda(int step, torch::Tensor img, torch::Tensor grad);
#endif

// Macros
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " was expected to be a CUDA tensor.")
#define CHECK_NOT_CUDA(x) AT_ASSERTM(!x.type().is_cuda(), #x " was not expected to be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// The interface function
void step(int step, torch::Tensor img, torch::Tensor grad) {
    CHECK_CONTIGUOUS(img);
    CHECK_CONTIGUOUS(grad);

#ifdef WITH_CUDA
    if (img.type().is_cuda()) {
        CHECK_CUDA(grad);
        step_cuda(step, img, grad);
    } else {
        CHECK_NOT_CUDA(grad);
        step_cpu(step, img, grad);
    }
#else
    CHECK_NOT_CUDA(img);
    CHECK_NOT_CUDA(grad);
    step_cpu(step, img, grad);
#endif
}

// PyBind11 definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step, "Reconstruction Step (CUDA)");
}
