#!/usr/bin/env python3

from src import GradientMap

from PIL import Image
import torch.cuda

import sys
import time

# Enable CUDA
cuda = False
if torch.cuda.is_available():
    dev = torch.device('cuda')
    if dev.type == 'cuda':
        cuda = True
        # Initializing CUDA shouldn't be done manually.
        # It is only done here for aesthetic reasons, as it would mess up
        # the timing of the prints.
        print("Initializing CUDA ... ", end=''); sys.stdout.flush()
        torch.Tensor([]).to(device=dev)
        print("done")
print("Device:", dev)

# Load Images
print("Opening images ... ", end=''); sys.stdout.flush()
bg = Image.open('img/bg_large.jpeg')
fg = Image.open('img/fg.png')
print("done")

# Compute Gradient Maps
print("Computing gradients ... ", end=''); sys.stdout.flush()
bg_grad = GradientMap.from_image(bg, device=dev)
fg_grad = GradientMap.from_image(fg, device=dev)
print("done")

# Insert foreground gradient into background
print("Merging gradients ... ", end=''); sys.stdout.flush()
bg_grad.paste_gradient(fg_grad, -20, -20)
print("done")

# COMPUTE
print("Reconstruction ... ", end=''); sys.stdout.flush()
num_iters = 1000

start = time.time()

bg_grad.reconstruct(num_iters)

if cuda:
    torch.cuda.synchronize()
duration = time.time() - start

print("done")
print("Speed:", int(100*num_iters/duration)/100, "iter/s")

# Show output image
bg_grad.get_image().show()
