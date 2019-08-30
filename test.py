#!/usr/bin/env python3

from src import GradientMap

from PIL import Image
import torch.cuda
import numpy as np

import sys

# Settings
num_iters = 1000

# Enable CUDA
dev = torch.device('cuda')
if dev is None or dev.type != 'cuda':
    print("No cuda device found. CUDA test is disabled.")
    use_cuda=False
else:
    use_cuda=True

# Load Images
print("Opening images ... ", end=''); sys.stdout.flush()
bg = Image.open('img/bg.png')
fg = Image.open('img/fg.png')
print("done")

# Compute CPU result
print("Computing CPU results ... ", end=''); sys.stdout.flush()
bg_grad = GradientMap.from_image(bg)
fg_grad = GradientMap.from_image(fg)
bg_grad.paste_gradient(fg_grad, -20, -20)
bg_grad.reconstruct(num_iters)
result_cpu = bg_grad.get_image()
print("done")

# Compute CUDA result
if use_cuda:
    print("Computing CUDA results ... ", end=''); sys.stdout.flush()
    bg_grad = GradientMap.from_image(bg, device=dev)
    fg_grad = GradientMap.from_image(fg, device=dev)
    bg_grad.paste_gradient(fg_grad, -20, -20)
    bg_grad.reconstruct(num_iters)
    result_cuda = bg_grad.get_image()
    print("done")

# Load Reference result
result_reference = Image.open('img/result_1000.png')

# Compare CPU and Reference
if not np.array_equal(np.asarray(result_reference), np.asarray(result_cpu)):
    print("CPU result differs from reference!")
    exit(1)
else:
    print("CPU result is correct.")

# Compare CUDA and CPU
if use_cuda:
    if not np.array_equal(np.asarray(result_cuda), np.asarray(result_cpu)):
        print("CPU and GPU results differ!")
        exit(1)
    else:
        print("CPU and GPU results are equal.")
