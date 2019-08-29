#!/usr/bin/env python3

from time import time
from PIL import Image

from src import GradientMap

import torch

# Enable CUDA
dev = None #torch.device('cuda')
print("Device:", dev)

# Load Images
bg = Image.open('img/bg_large.jpeg')
fg = Image.open('img/fg.jpg')

# Compute Gradient Maps
bg_grad = GradientMap.from_image(bg, device=dev)
fg_grad = GradientMap.from_image(fg, device=dev)

# Insert foreground gradient into background
bg_grad.paste_gradient(fg_grad, 10, 10)

# COMPUTE
print("Computing ...", end = '')
start = time()

num_iters = 100
bg_grad.reconstruct(num_iters)
img = bg_grad.get_image()

duration = time() - start
print("done")
print("Speed:", num_iters/duration, "iter/s")
# COMPUTE DONE

# Show output image
img.show()
