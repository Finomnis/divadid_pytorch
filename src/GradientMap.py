from torchvision import transforms
import torch.nn.functional
from torch.utils.cpp_extension import load

import numpy as np

import os
import sys

# Load external cuda file
script_path = os.path.dirname(os.path.realpath(__file__))
print("Loading compute kernels ... ", end='')
sys.stdout.flush()
ext_cuda = load(name='ext_cuda', extra_cflags=['-O3'], sources=[
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

        # README: I am not entirely sure if the cuda implementation is exactly equivalent to the original, needs review

        # Extract and pad
        grad_other_x = torch.nn.functional.pad(other.grad_x, [0, 1, 0, 0])
        grad_other_y = torch.nn.functional.pad(other.grad_y, [0, 0, 0, 1])
        assert(grad_other_x.shape == grad_other_y.shape)

        # Cut grad_other for the first time, if x < 0 or y < 0
        if x < 0:
            grad_other_x = grad_other_x[:, :, -x:]
            grad_other_y = grad_other_y[:, :, -x:]
            x = 0
        if y < 0:
            grad_other_x = grad_other_x[:, -y:, :]
            grad_other_y = grad_other_y[:, -y:, :]
            y = 0

        # Cut grad_self to fit grad_other
        grad_self_x = self.grad_x[:, y:y + grad_other_x.size(1), x:x + grad_other_x.size(2)]
        grad_self_y = self.grad_y[:, y:y + grad_other_x.size(1), x:x + grad_other_x.size(2)]

        # Cut grad_self with itself to ensure they are identical in size.
        # Possible, because they are both missing one row or column.
        grad_self_x = grad_self_x[:, :grad_self_y.size(1), :grad_self_y.size(2)]
        grad_self_y = grad_self_y[:, :grad_self_x.size(1), :grad_self_x.size(2)]
        assert(grad_self_x.shape == grad_self_y.shape)

        # If grad_self is a different size than grad_other, this means we are on the right or bottom edge.
        # Then, cut grad_other again, this time on the right and bottom side
        grad_other_x = grad_other_x[:, :grad_self_y.size(1), :grad_self_x.size(2)]
        grad_other_y = grad_other_y[:, :grad_self_y.size(1), :grad_self_x.size(2)]
        assert (grad_other_x.shape == grad_other_y.shape)
        assert (grad_other_x.shape == grad_self_x.shape)
        assert (grad_other_y.shape == grad_self_y.shape)

        # Apply boosting
        grad_other_x *= boost
        grad_other_y *= boost

        # Compute replacement condition
        grad_self_norm = grad_self_x ** 2 + grad_self_y ** 2
        grad_other_norm = grad_other_x ** 2 + grad_other_y ** 2
        replacement_stencil = grad_other_norm > grad_self_norm

        # Replace
        grad_self_x[replacement_stencil] = grad_other_x[replacement_stencil]
        grad_self_y[replacement_stencil] = grad_other_y[replacement_stencil]

    def reconstruct(self, steps):
        for i in range(steps):
            #import time
            #start = time.time()
            if i % 2 == 0:
                ext_cuda.step(i, self.img, self.grad_x)
            else:
                ext_cuda.step(i, self.img, self.grad_y)
            #duration = time.time() - start
            #print(i%4, ":", int(duration*1e6*100)/100)

