#!/usr/bin/env python3

from src import GradientMap

from PIL import Image
import torch.cuda
from torchvision import transforms
import numpy as np

import sys

# Settings
num_iters = 1000

# Enable CUDA
has_cuda = False
dev = None
if torch.cuda.is_available():
    dev = torch.device('cuda')
    if dev.type == 'cuda':
        has_cuda = True

if not has_cuda:
    print("No cuda device found. CUDA test is disabled.")

# Load Images
print("Opening images ... ", end=''); sys.stdout.flush()
# Foreground and background are switched. Combined with a negative offset, the pasted gradients overlap on all sides.
# This tests all special cases I could think off. (eg. unrolling and border handling)
fg = Image.open('img/bg.png')
bg = Image.open('img/fg.png')
print("done")

# Compute CPU result
print("Computing CPU results ... ", end=''); sys.stdout.flush()
bg_grad = GradientMap.from_image(bg)
fg_grad = GradientMap.from_image(fg)
bg_grad.paste_gradient(fg_grad, -30, -30)
bg_grad.reconstruct(num_iters)
result_cpu = bg_grad.get_image()
print("done")

# Compute CUDA result
if has_cuda:
    print("Computing CUDA results ... ", end=''); sys.stdout.flush()
    bg_grad = GradientMap.from_image(bg, device=dev)
    fg_grad = GradientMap.from_tensor(transforms.ToTensor()(np.asarray(fg)).to(device=dev))
    bg_grad.paste_gradient(fg_grad, -30, -30)
    bg_grad.reconstruct(num_iters)
    result_cuda = bg_grad.get_image()
    print("done")

# Load Reference result
result_reference = Image.open('img/result_1000.png')

# Init exit state
exit_state = 0
print()

# Compare CPU and Reference
if not np.array_equal(np.asarray(result_reference), np.asarray(result_cpu)):
    print("ERROR: CPU result differs from reference!")
    exit_state = 1
else:
    print("CPU result is correct.")

# result_cuda.show()
# result_cpu.show()

# Compare CUDA and CPU
if has_cuda:
    if not np.array_equal(np.asarray(result_cuda), np.asarray(result_reference)):
        print("ERROR: GPU result differs from reference!")
        exit_state = 1
    else:
        print("GPU result is correct.")

    if not np.array_equal(np.asarray(result_cuda), np.asarray(result_cpu)):
        print("ERROR: CPU and GPU results differ!")
        exit_state = 1
    else:
        print("CPU and GPU results are equal.")

exit(exit_state)
