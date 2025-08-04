import mpmath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpmath.mp.dps = 50
exponents = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 69794, 86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933, 136279841]
exponents = np.array(exponents, dtype=float)

unfolded = []
for p in exponents:
    if p > 2 * mpmath.pi * mpmath.exp(1):  # Avoid log issues for small p
        log_term = mpmath.log(p / (2 * mpmath.pi * mpmath.exp(1)))
        unfolded_p = p / (2 * mpmath.pi * log_term)
        unfolded.append(float(unfolded_p))
    else:
        unfolded.append(float(p))  # Raw for small exponents
unfolded = np.array(unfolded)

phi = (1 + np.sqrt(5)) / 2
theta = 2 * np.pi * unfolded / phi
x = unfolded * np.cos(theta)
y = unfolded * np.sin(theta)
z = unfolded  # Linear ascent for helical structure

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='green', marker='o', s=20, label='Mersenne Exponents')
ax.plot(x, y, z, c='purple', linewidth=1, label='Helical Geodesic')
ax.set_xlabel('X (Cosine Projection)')
ax.set_ylabel('Y (Sine Projection)')
ax.set_zlabel('Z (Unfolded Height)')
ax.set_title('3D Helical Embedding of Mersenne Prime Exponents')
ax.legend()
ax.view_init(elev=20, azim=45)  # Optimal frame adjustment
plt.show()

