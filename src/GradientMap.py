from torchvision import transforms
import torch.nn.functional
from torch.utils.cpp_extension import load

import numpy as np

import os
import time

# Load external cuda file
script_path = os.path.dirname(os.path.realpath(__file__))
ext_cuda = load(name='ext_cuda', sources=[
    os.path.join(script_path, 'cuda', 'GradientMap.cpp'),
    os.path.join(script_path, 'cuda', 'GradientMap.cu'),
])


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
        def step_lr():
            ext_cuda.step(i, self.img, self.grad_x)
            # for x in range(1, self.img.size(2) - 1):
            #    self.img[:, 1:-1, x] = (self.img[:, 1:-1, x] + self.img[:, 1:-1, x - 1] + self.grad_x[:, 1:-1, x - 1]) / 2

        def step_rl():
            ext_cuda.step(i, self.img, self.grad_x)

            #for x in range(self.img.size(2) - 2, 0, -1):
            #    self.img[:, 1:-1, x] = (self.img[:, 1:-1, x] + self.img[:, 1:-1, x + 1] - self.grad_x[:, 1:-1, x]) / 2

        def step_tb():
            for y in range(1, self.img.size(1) - 1):
                self.img[:, y, 1:-1] = (self.img[:, y, 1:-1] + self.img[:, y - 1, 1:-1] + self.grad_y[:, y - 1,
                                                                                          1:-1]) / 2

        def step_bt():
            for y in range(self.img.size(1) - 2, 0, -1):
                self.img[:, y, 1:-1] = (self.img[:, y, 1:-1] + self.img[:, y + 1, 1:-1] - self.grad_y[:, y, 1:-1]) / 2

        timestamps = []
        for i in range(steps):
            start = time.time()
            if i % 4 == 0:
                step_lr()
            elif i % 4 == 1:
                step_tb()
            elif i % 4 == 2:
                step_rl()
            else:
                step_bt()
            timestamps.append(int(10000 * (time.time() - start)))
            if i % 4 == 3:
                print(timestamps)
                timestamps = []
