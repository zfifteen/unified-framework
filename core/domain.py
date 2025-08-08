from abc import ABC
import collections
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import divisors, isprime
import mpmath as mp
import numpy as np

mp.mp.dps = 50  # High precision for large n and modular ops

PHI = (1 + mp.sqrt(5)) / 2
E_SQUARED = mp.exp(2)

class UniversalZetaShift(ABC):
    """
    Abstract base class for universal zeta shift calculations with memoization.
    
    This class provides computed getters that benefit from automatic caching,
    ensuring O(1) retrieval on subsequent calls without changing external behavior.
    
    Example:
        >>> uzz = UniversalZetaShift(2, 3, 5)
        >>> # First calls populate cache
        >>> d1 = uzz.getD()  # Computed and cached
        >>> d2 = uzz.getD()  # Retrieved from cache
        >>> assert d1 == d2  # Identical results
        >>> 
        >>> # Cache inspection (internal use)
        >>> print(len(uzz._cache))  # Shows number of cached values
    """
    def __init__(self, a, b, c):
        if a == 0 or b == 0 or c == 0:
            raise ValueError("Parameters cannot be zero.")
        self.a = mp.mpmathify(a)
        self.b = mp.mpmathify(b)
        self.c = mp.mpmathify(c)
        self._cache = {}

    def compute_z(self):
        if 'z' in self._cache:
            return self._cache['z']
        try:
            result = self.a * (self.b / self.c)
        except ZeroDivisionError:
            result = mp.inf
        self._cache['z'] = result
        return result

    def getD(self):
        if 'D' in self._cache:
            return self._cache['D']
        try:
            result = self.c / self.a
        except ZeroDivisionError:
            result = mp.inf
        self._cache['D'] = result
        return result

    def getE(self):
        if 'E' in self._cache:
            return self._cache['E']
        try:
            result = self.c / self.b
        except ZeroDivisionError:
            result = mp.inf
        self._cache['E'] = result
        return result

    def getF(self):
        if 'F' in self._cache:
            return self._cache['F']
        try:
            d_over_e = self.getD() / self.getE()
            result = PHI * ((d_over_e % PHI) / PHI) ** mp.mpf(0.3)
        except ZeroDivisionError:
            result = mp.inf
        self._cache['F'] = result
        return result

    def getG(self):
        if 'G' in self._cache:
            return self._cache['G']
        try:
            f = self.getF()
            result = (self.getE() / f) / E_SQUARED
        except ZeroDivisionError:
            result = mp.inf
        self._cache['G'] = result
        return result

    def getH(self):
        if 'H' in self._cache:
            return self._cache['H']
        try:
            result = self.getF() / self.getG()
        except ZeroDivisionError:
            result = mp.inf
        self._cache['H'] = result
        return result

    def getI(self):
        if 'I' in self._cache:
            return self._cache['I']
        try:
            g_over_h = self.getG() / self.getH()
            result = PHI * ((g_over_h % PHI) / PHI) ** mp.mpf(0.3)
        except ZeroDivisionError:
            result = mp.inf
        self._cache['I'] = result
        return result

    def getJ(self):
        if 'J' in self._cache:
            return self._cache['J']
        try:
            result = self.getH() / self.getI()
        except ZeroDivisionError:
            result = mp.inf
        self._cache['J'] = result
        return result

    def getK(self):
        if 'K' in self._cache:
            return self._cache['K']
        try:
            result = (self.getI() / self.getJ()) / E_SQUARED
        except ZeroDivisionError:
            result = mp.inf
        self._cache['K'] = result
        return result

    def getL(self):
        if 'L' in self._cache:
            return self._cache['L']
        try:
            result = self.getJ() / self.getK()
        except ZeroDivisionError:
            result = mp.inf
        self._cache['L'] = result
        return result

    def getM(self):
        if 'M' in self._cache:
            return self._cache['M']
        try:
            k_over_l = self.getK() / self.getL()
            result = PHI * ((k_over_l % PHI) / PHI) ** mp.mpf(0.3)
        except ZeroDivisionError:
            result = mp.inf
        self._cache['M'] = result
        return result

    def getN(self):
        if 'N' in self._cache:
            return self._cache['N']
        try:
            result = self.getL() / self.getM()
        except ZeroDivisionError:
            result = mp.inf
        self._cache['N'] = result
        return result

    def getO(self):
        if 'O' in self._cache:
            return self._cache['O']
        try:
            result = self.getM() / self.getN()
        except ZeroDivisionError:
            result = mp.inf
        self._cache['O'] = result
        return result

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

    def get_helical_coordinates(self, r_normalized=1.0):
        """
        Get helical embedding coordinates following Task 3 specifications:
        - θ_D = 2*π*n/50
        - x = r*cos(θ_D), y = r*sin(θ_D), z = n
        - w = I, u = O from zeta chains
        """
        attrs = self.attributes
        n = float(self.a)
        theta_D = 2 * mp.pi * n / 50
        
        x = r_normalized * mp.cos(theta_D)
        y = r_normalized * mp.sin(theta_D)
        z = n
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