from abc import ABC
import collections
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import divisors, isprime
import mpmath as mp

mp.mp.dps = 50  # High precision for large n and modular ops

PHI = (1 + mp.sqrt(5)) / 2
E_SQUARED = mp.exp(2)

class UniversalZetaShift(ABC):
    def __init__(self, a, b, c):
        if a == 0 or b == 0 or c == 0:
            raise ValueError("Parameters cannot be zero.")
        self.a = mp.mpmathify(a)
        self.b = mp.mpmathify(b)
        self.c = mp.mpmathify(c)

    def compute_z(self):
        try:
            return self.a * (self.b / self.c)
        except ZeroDivisionError:
            return mp.inf

    def getD(self):
        try:
            return self.c / self.a
        except ZeroDivisionError:
            return mp.inf

    def getE(self):
        try:
            return self.c / self.b
        except ZeroDivisionError:
            return mp.inf

    def getF(self):
        try:
            d_over_e = self.getD() / self.getE()
            return PHI * ((d_over_e % PHI) / PHI) ** mp.mpf(0.3)
        except ZeroDivisionError:
            return mp.inf

    def getG(self):
        try:
            f = self.getF()
            return (self.getE() / f) / E_SQUARED
        except ZeroDivisionError:
            return mp.inf

    def getH(self):
        try:
            return self.getF() / self.getG()
        except ZeroDivisionError:
            return mp.inf

    def getI(self):
        try:
            g_over_h = self.getG() / self.getH()
            return PHI * ((g_over_h % PHI) / PHI) ** mp.mpf(0.3)
        except ZeroDivisionError:
            return mp.inf

    def getJ(self):
        try:
            return self.getH() / self.getI()
        except ZeroDivisionError:
            return mp.inf

    def getK(self):
        try:
            return (self.getI() / self.getJ()) / E_SQUARED
        except ZeroDivisionError:
            return mp.inf

    def getL(self):
        try:
            return self.getJ() / self.getK()
        except ZeroDivisionError:
            return mp.inf

    def getM(self):
        try:
            k_over_l = self.getK() / self.getL()
            return PHI * ((k_over_l % PHI) / PHI) ** mp.mpf(0.3)
        except ZeroDivisionError:
            return mp.inf

    def getN(self):
        try:
            return self.getL() / self.getM()
        except ZeroDivisionError:
            return mp.inf

    def getO(self):
        try:
            return self.getM() / self.getN()
        except ZeroDivisionError:
            return mp.inf

    @property
    def attributes(self):
        return {
            'a': self.a, 'b': self.b, 'c': self.c, 'z': self.compute_z(),
            'D': self.getD(), 'E': self.getE(), 'F': self.getF(), 'G': self.getG(),
            'H': self.getH(), 'I': self.getI(), 'J': self.getJ(), 'K': self.getK(),
            'L': self.getL(), 'M': self.getM(), 'N': self.getN(), 'O': self.getO()
        }

class DiscreteZetaShift(UniversalZetaShift):
    def __init__(self, n, v=1.0, delta_max=E_SQUARED):
        self.vortex = collections.deque()  # Instance-level vortex
        n = mp.mpmathify(n)
        d_n = len(divisors(int(n)))  # sympy for divisors, cast to int if needed
        kappa = d_n * mp.log(n + 1) / E_SQUARED
        delta_n = v * kappa
        super().__init__(a=n, b=delta_n, c=delta_max)
        self.v = v
        self.f = round(float(self.getG()))  # Cast to float for rounding
        self.w = round(float(2 * mp.pi / PHI))

        self.vortex.append(self)
        while len(self.vortex) > self.f:
            self.vortex.popleft()

    def unfold_next(self):
        successor = DiscreteZetaShift(self.a + 1, v=self.v, delta_max=self.c)
        self.vortex.append(successor)
        while len(self.vortex) > successor.f:
            self.vortex.popleft()
        return successor

    def get_3d_coordinates(self):
        attrs = self.attributes
        theta_d = PHI * ((attrs['D'] % PHI) / PHI) ** mp.mpf(0.3)
        theta_e = PHI * ((attrs['E'] % PHI) / PHI) ** mp.mpf(0.3)
        x = self.a * mp.cos(theta_d)
        y = self.a * mp.sin(theta_e)
        z = attrs['F'] / E_SQUARED
        return (float(x), float(y), float(z))

    def get_4d_coordinates(self):
        attrs = self.attributes
        x, y, z = self.get_3d_coordinates()
        t = -self.c * (attrs['O'] / PHI)
        return (float(t), x, y, z)

    def get_5d_coordinates(self):
        attrs = self.attributes
        theta_d = PHI * ((attrs['D'] % PHI) / PHI) ** mp.mpf(0.3)
        theta_e = PHI * ((attrs['E'] % PHI) / PHI) ** mp.mpf(0.3)
        x = self.a * mp.cos(theta_d)
        y = self.a * mp.sin(theta_e)
        z = attrs['F'] / E_SQUARED
        w = attrs['I']
        u = attrs['O']
        return (float(x), float(y), float(z), float(w), float(u))

    @classmethod
    def generate_key(cls, N, seed_n=2):
        zeta = cls(seed_n)
        trajectory_o = [zeta.getO()]
        for _ in range(1, N):
            zeta = zeta.unfold_next()
            trajectory_o.append(zeta.getO())
        hash_input = ''.join(mp.nstr(o, 20) for o in trajectory_o)  # Higher precision
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

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
        is_primes = np.array([isprime(int(shift.a)) for shift in shifts])  # Cast to int
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
print("Vortex O values:", [float(inst.getO()) for inst in zeta.vortex])  # Instance vortex
sample_key = DiscreteZetaShift.generate_key(10)
print("Sample generated key:", sample_key)