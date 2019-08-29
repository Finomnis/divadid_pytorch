from torchvision import transforms
import torch.nn.functional
from torch.utils.cpp_extension import load

import numpy as np

import os
import time

# Load external cuda file
script_path = os.path.dirname(os.path.realpath(__file__))
print("Loading compute kernels ... ", end = '')
ext_cuda = load(name='ext_cuda', sources=[
    os.path.join(script_path, 'cuda', 'GradientMap.cpp'),
    os.path.join(script_path, 'cuda', 'GradientMap_cuda.cu'),
    os.path.join(script_path, 'cuda', 'GradientMap_cpu.cpp'),
])
print("done")


class GradientMap:

    def __init__(self, img, grad_x, grad_y):
        self.img = img
        self.grad_x = grad_x
        self.grad_y = grad_y

    @staticmethod
    def from_image(img, channels=(0, 1, 2), device=None):
        img = transforms.ToTensor()(np.asarray(img))

        if device:
            img = img.to(device=device)

        img = img[channels, :, :]
        if len(img.shape) < 3:
            img = img.unsqueeze(0)

        grad_x = img[:, :, 1:] - img[:, :, :-1]
        grad_y = img[:, 1:, :] - img[:, :-1, :]

        return GradientMap(img, grad_x, grad_y)

    def get_image(self):
        return transforms.ToPILImage()(self.img.cpu().clamp(0, 1))

    def paste_gradient(self, other, x, y, boost=1.0):
        """Inserts a given gradient at position x,y into self."""

        # Not entirely sure if that implementation is exactly equivalent to the original, needs review
        _, size_x, size_y = other.img.shape

        # Cut.
        # TODO: Implement. Currently relies on the fact that the pasted image is completely inside our image
        grad_self_x = self.grad_x[:, x:x + size_x, y:y + size_y]
        grad_self_y = self.grad_y[:, x:x + size_x, y:y + size_y]
        grad_other_x = other.grad_x
        grad_other_y = other.grad_y

        # Apply boosting
        grad_other_x *= boost
        grad_other_y *= boost

        # Pad
        grad_other_x = torch.nn.functional.pad(grad_other_x, (0, 1, 0, 0))
        grad_other_y = torch.nn.functional.pad(grad_other_y, (0, 0, 0, 1))

        # Compute replacement condition
        grad_self_norm = grad_self_x ** 2 + grad_self_y ** 2
        grad_other_norm = grad_other_x ** 2 + grad_other_y ** 2
        replacement_stencil = grad_other_norm > grad_self_norm

        # Replace
        grad_self_x[replacement_stencil] = grad_other_x[replacement_stencil]
        grad_self_y[replacement_stencil] = grad_other_y[replacement_stencil]

    def reconstruct(self, steps):
        for i in range(steps):
            if i % 2 == 0:
                ext_cuda.step(i, self.img, self.grad_x)
            else:
                ext_cuda.step(i, self.img, self.grad_y)

