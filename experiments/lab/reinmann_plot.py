import mpmath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpmath.mp.dps = 50
J = 500  # Number of zeros
zeros = [mpmath.zetazero(j+1) for j in range(J)]  # First J zeros, imag parts positive
imag_zeros = np.array([float(z.imag) for z in zeros])

unfolded = []
for t in imag_zeros:
    log_term = mpmath.log(t / (2 * mpmath.pi * mpmath.exp(1)))
    unfolded_t = t / (2 * mpmath.pi * log_term)
    unfolded.append(float(unfolded_t))
unfolded = np.array(unfolded)

phi = (1 + np.sqrt(5)) / 2
theta = 2 * np.pi * unfolded / phi
x = unfolded * np.cos(theta)
y = unfolded * np.sin(theta)
z = unfolded  # Linear ascent along z for helical rise

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', marker='o', s=20, label='Riemann Zeros')
ax.plot(x, y, z, c='red', linewidth=1, label='Helical Geodesic')
ax.set_xlabel('X (Cosine Projection)')
ax.set_ylabel('Y (Sine Projection)')
ax.set_zlabel('Z (Unfolded Height)')
ax.set_title('3D Helical Embedding of Riemann Zeros')
ax.legend()
ax.view_init(elev=20, azim=45)  # Adjust for optimal frame
plt.show()