import numpy as np
from deconvolution import compute_psnr, wiener_filtering, gaussian_kernel
import os

# Загрузка изображений
original = np.load('examples/original.npy')
noisy = np.load('examples/noisy.npy')

kernel = gaussian_kernel(15, 5)
filtered = wiener_filtering(noisy, kernel)

psnr_noisy = compute_psnr(noisy, original)
psnr_filtered = compute_psnr(filtered, original)

print(f"PSNR noisy: {psnr_noisy:.2f}")
print(f"PSNR filtered: {psnr_filtered:.2f}")
print(f"Improvement: {psnr_filtered - psnr_noisy:.2f}")