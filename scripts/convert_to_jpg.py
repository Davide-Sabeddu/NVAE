import numpy as np
import matplotlib.pyplot as plt
import os

# Load all samples
samples = []
for i in range(500):
    f = np.load(f'/tmp/expr/gpu_0_samples_{i}.npz')
    samples.append(f[f.files[0]])  # grab the array inside

samples = np.concatenate(samples, axis=0)
print("Samples shape:", samples.shape)

# Save a grid of the first 64
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    img = samples[i]
    # MNIST is grayscale, shape likely (1, 28, 28) or (28, 28)
    if img.ndim == 3:
        img = img.squeeze(0)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig('/tmp/expr/generated_samples_grid.png', dpi=150)
print("Saved grid to /tmp/expr/generated_samples_grid.png")