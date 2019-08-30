#!/usr/bin/env python3

from time import time
from PIL import Image

from src import GradientMap

import torch.cuda
import sys


# Enable CUDA
dev = None #torch.device('cuda')
print("Device:", dev)
if dev is not None and dev.type == 'cuda':
    print("Initializing CUDA ... ", end=''); sys.stdout.flush()
    # Initializing CUDA shouldn't be done manually. It is only done here for aesthetic reasons, as it would mess up
    # the timing of the prints.
    torch.Tensor([]).to(device=dev)
    print("done")

# Load Images
print("Opening images ... ", end=''); sys.stdout.flush()
bg = Image.open('img/bg_large.jpeg')
fg = Image.open('img/fg.jpg')
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
start = time()

num_iters = 10000
bg_grad.reconstruct(num_iters)
img = bg_grad.get_image()

duration = time() - start
print("done")
print("Speed:", int(100*num_iters/duration)/100, "iter/s")

# Show output image
img.show()
