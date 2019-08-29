#!/usr/bin/env python3

from time import time
from PIL import Image

from src import GradientMap

import torch

dev = torch.device('cuda')
print(dev)

bg = Image.open('img/bg_large.jpeg')
fg = Image.open('img/fg.jpg')

bg_grad = GradientMap.from_image(bg, device=dev)
fg_grad = GradientMap.from_image(fg, device=dev)


bg_grad.paste_gradient(fg_grad, 0, 0)

num_iters = 100

start = time()
bg_grad.reconstruct(num_iters)
duration = time() - start

print("Speed:", num_iters/duration, "iter/s")

bg_grad.get_image().show()
