import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def kappa(n):
    d_n = len([i for i in range(1, n + 1) if n % i == 0])
    return d_n * np.log(n + 1) / np.exp(2)


N = 500
nums = np.arange(2, N + 1)
kappas = np.array([kappa(n) for n in nums])
is_primes = np.array([is_prime(n) for n in nums])

# 3D scatter data
x = nums
y = kappas
z = np.zeros_like(x)
z[is_primes] = 1  # mark primes differently

# Helical coordinates
theta = 2 * np.pi * nums / 50
r = kappas / max(kappas)
x_helix = r * np.cos(theta)
y_helix = r * np.sin(theta)
z_helix = nums

# Set up figure
fig = plt.figure(figsize=(18, 5))

# Plot 1: 3D Prime/Composite Scatter in (n, κ, prime_flag)
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(x[~is_primes], y[~is_primes], z[~is_primes], c='gray', label='Composite', alpha=0.5)
ax1.scatter(x[is_primes], y[is_primes], z[is_primes], c='red', label='Prime')
ax1.set_title("3D Scatter (n, κ(n), PrimeFlag)")
ax1.set_xlabel("n")
ax1.set_ylabel("κ(n)")
ax1.set_zlabel("Prime Flag")
ax1.legend()

# Plot 2: 3D Curvature Ridge
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(x, y, zs=0, zdir='z', label='κ(n)', color='blue')
ax2.set_title("3D Curvature Ridge")
ax2.set_xlabel("n")
ax2.set_ylabel("κ(n)")
ax2.set_zlabel("z")

# Plot 3: Helical κ(n) Projection
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(x_helix, y_helix, z_helix, label='κ(n) helix', color='purple')
ax3.scatter(x_helix[is_primes], y_helix[is_primes], z_helix[is_primes], c='red', label='Prime')
ax3.set_title("Helical Projection of κ(n)")
ax3.set_xlabel("x (κ*cosθ)")
ax3.set_ylabel("y (κ*sinθ)")
ax3.set_zlabel("n")
ax3.legend()

plt.tight_layout()
plt.show()
