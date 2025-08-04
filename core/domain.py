from abc import ABC
import math
import numpy as np
from sympy import divisors, isprime
import collections
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PHI = (1 + math.sqrt(5)) / 2
E_SQUARED = np.exp(2)

class UniversalZetaShift(ABC):
    def __init__(self, a, b, c):
        if a == 0 or b == 0 or c == 0:
            raise ValueError("Parameters cannot be zero.")
        self.a = a
        self.b = b
        self.c = c

    def compute_z(self):
        return self.a * (self.b / self.c)

    def getD(self):
        return self.c / self.a

    def getE(self):
        return self.c / self.b

    def getF(self):
        d_over_e = self.getD() / self.getE()
        return PHI * ((d_over_e % PHI) / PHI) ** 0.3

    def getG(self):
        f = self.getF()
        return (self.getE() / f) / E_SQUARED

    def getH(self):
        return self.getF() / self.getG()

    def getI(self):
        g_over_h = self.getG() / self.getH()
        return PHI * ((g_over_h % PHI) / PHI) ** 0.3

    def getJ(self):
        return self.getH() / self.getI()

    def getK(self):
        return (self.getI() / self.getJ()) / E_SQUARED

    def getL(self):
        return self.getJ() / self.getK()

    def getM(self):
        k_over_l = self.getK() / self.getL()
        return PHI * ((k_over_l % PHI) / PHI) ** 0.3

    def getN(self):
        return self.getL() / self.getM()

    def getO(self):
        return self.getM() / self.getN()

    @property
    def attributes(self):
        return {
            'a': self.a, 'b': self.b, 'c': self.c, 'z': self.compute_z(),
            'D': self.getD(), 'E': self.getE(), 'F': self.getF(), 'G': self.getG(),
            'H': self.getH(), 'I': self.getI(), 'J': self.getJ(), 'K': self.getK(),
            'L': self.getL(), 'M': self.getM(), 'N': self.getN(), 'O': self.getO()
        }

class DiscreteZetaShift(UniversalZetaShift):
    vortex = collections.deque()  # Shared FIFO vortex, unlimited

    def __init__(self, n, v=1.0, delta_max=E_SQUARED):
        d_n = len(divisors(n))
        kappa = d_n * math.log(n + 1) / E_SQUARED
        delta_n = v * kappa
        super().__init__(a=n, b=delta_n, c=delta_max)
        self.v = v
        self.f = round(self.getG())  # Derive f ≈ π via G
        self.w = round(2 * math.pi / PHI)  # Derive w ≈ π via helical phase

        # Append self to vortex, then limit vortex length to self.f
        self.vortex.append(self)
        # If vortex is longer than desired max (self.f), drop oldest
        while len(self.vortex) > self.f:
            self.vortex.popleft()

    def unfold_next(self):
        successor = DiscreteZetaShift(self.a + 1, v=self.v, delta_max=self.c)
        self.vortex.append(successor)
        # Ensure vortex length does not exceed the new successor's f
        while len(self.vortex) > successor.f:
            self.vortex.popleft()
        return successor

    def get_3d_coordinates(self):
        attrs = self.attributes
        theta_d = PHI * ((attrs['D'] % PHI) / PHI) ** 0.3
        theta_e = PHI * ((attrs['E'] % PHI) / PHI) ** 0.3
        x = self.a * math.cos(theta_d)
        y = self.a * math.sin(theta_e)
        z = attrs['F'] / E_SQUARED
        return (x, y, z)

    def get_4d_coordinates(self):
        attrs = self.attributes
        x, y, z = self.get_3d_coordinates()
        t = -self.c * (attrs['O'] / PHI)  # Time-like component for Minkowski-like bounding
        return (t, x, y, z)

    def get_5d_coordinates(self):
        attrs = self.attributes
        theta_d = PHI * ((attrs['D'] % PHI) / PHI) ** 0.3
        theta_e = PHI * ((attrs['E'] % PHI) / PHI) ** 0.3
        x = self.a * math.cos(theta_d)
        y = self.a * math.sin(theta_e)
        z = attrs['F'] / E_SQUARED
        w = attrs['I']
        u = attrs['O']
        return (x, y, z, w, u)

    @classmethod
    def generate_key(cls, N, seed_n=2):
        zeta = DiscreteZetaShift(seed_n)
        trajectory_o = [zeta.getO()]
        for _ in range(1, N):
            zeta = zeta.unfold_next()
            trajectory_o.append(zeta.getO())
        hash_input = ''.join(f"{o:.3f}" for o in trajectory_o)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]  # 256-bit key truncated

    @classmethod
    def get_coordinates_array(cls, dim=3, N=100, seed=2, v=1.0, delta_max=E_SQUARED):
        zeta = cls(seed, v, delta_max)
        shifts = [zeta]
        for _ in range(1, N):
            zeta = zeta.unfold_next()
            shifts.append(zeta)
        if dim == 3:
            coords = np.array([shift.get_3d_coordinates() for shift in shifts])
        elif dim == 4:
            coords = np.array([shift.get_4d_coordinates() for shift in shifts])
        else:
            raise ValueError("dim must be 3 or 4")
        is_primes = np.array([isprime(shift.a) for shift in shifts])
        return coords, is_primes

    @classmethod
    def plot_3d(cls, N=100, seed=2, v=1.0, delta_max=E_SQUARED, ax=None):
        coords, is_primes = cls.get_coordinates_array(dim=3, N=N, seed=seed, v=v, delta_max=delta_max)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[~is_primes, 0], coords[~is_primes, 1], coords[~is_primes, 2], c='b', label='Composites')
        ax.scatter(coords[is_primes, 0], coords[is_primes, 1], coords[is_primes, 2], c='r', label='Primes')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        return ax

    @classmethod
    def plot_4d_as_3d_with_color(cls, N=100, seed=2, v=1.0, delta_max=E_SQUARED, ax=None):
        coords, is_primes = cls.get_coordinates_array(dim=4, N=N, seed=seed, v=v, delta_max=delta_max)
        t, x, y, z = coords.T
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=t, cmap='viridis')
        plt.colorbar(scatter, label='Time-like t')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax

# Demonstration: Unfold to N=10, print vortex O values, generate sample key
zeta = DiscreteZetaShift(2)
for _ in range(9):
    zeta = zeta.unfold_next()
print("Vortex O values:", [inst.getO() for inst in DiscreteZetaShift.vortex])
sample_key = DiscreteZetaShift.generate_key(10)
print("Sample generated key:", sample_key)