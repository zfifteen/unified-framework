from abc import ABC
import collections
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import divisors, isprime
import mpmath as mp
import numpy as np

mp.mp.dps = 50  # High precision for large n and modular ops

# Import system instruction for compliance validation
try:
    from .system_instruction import enforce_system_instruction, get_system_instruction
    _SYSTEM_INSTRUCTION_AVAILABLE = True
except ImportError:
    _SYSTEM_INSTRUCTION_AVAILABLE = False
    # Fallback no-op decorator if system instruction not available
    def enforce_system_instruction(func):
        return func

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
    """
    Discrete domain implementation of Z Framework with system instruction compliance.
    
    Implements Z = n(Δ_n/Δ_max) where:
    - n: frame-dependent integer
    - Δ_n: measured frame shift κ(n) = d(n) · ln(n+1)/e²  
    - Δ_max: maximum shift bounded by e² or φ
    
    SYSTEM INSTRUCTION COMPLIANCE:
    - Follows discrete domain form Z = n(Δ_n/Δ_max)
    - Uses e² normalization for variance minimization
    - Implements curvature formula κ(n) = d(n) · ln(n+1)/e²
    - Provides 5D helical embeddings for geometric analysis
    """
    
    @enforce_system_instruction
    def __init__(self, n, v=1.0, delta_max=E_SQUARED):
        self.vortex = collections.deque()  # Instance-level vortex
        n = mp.mpmathify(n)
        d_n = len(divisors(int(n)))  # sympy for divisors, cast to int if needed
        
        # Enhanced discrete curvature κ(n) = d(n) · ln(n+1)/e² with proper bounds
        kappa = d_n * mp.log(n + 1) / E_SQUARED
        
        # Apply bounds: κ(n) bounded by e² or φ for numerical stability
        kappa_bounded = min(kappa, E_SQUARED, PHI)
        
        # Discrete domain: Z = n(Δ_n/Δ_max) where Δ_n = v * κ(n)
        delta_n = v * kappa_bounded
        
        # Store unbounded kappa for analysis
        self.kappa_raw = kappa
        self.kappa_bounded = kappa_bounded
        self.delta_n = delta_n
        
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

    def get_curvature_geodesic_parameter(self):
        """
        Compute curvature-based geodesic parameter k(n) to replace hardcoded ratios.
        Uses bounded curvature κ(n) to derive optimal k for minimal variance.
        
        Strategy: Use variance-minimizing transformation based on empirical analysis.
        """
        # Normalize κ(n) relative to its expected scale
        kappa_norm = float(self.kappa_bounded) / float(PHI)  # Use φ as normalizing constant
        
        # Variance-minimizing function derived from optimization
        # k(κ) = 0.118 + 0.382 * exp(-2.0 * κ_norm) for low variance
        k_geodesic = 0.118 + 0.382 * mp.exp(-2.0 * kappa_norm)
        
        # Ensure k stays in stable range [0.05, 0.5]
        k_geodesic = max(0.05, min(0.5, float(k_geodesic)))
        
        return k_geodesic

    def get_3d_coordinates(self):
        attrs = self.attributes
        k_geo = self.get_curvature_geodesic_parameter()
        theta_d = PHI * ((attrs['D'] % PHI) / PHI) ** mp.mpf(k_geo)
        theta_e = PHI * ((attrs['E'] % PHI) / PHI) ** mp.mpf(k_geo)
        
        # Apply variance-minimizing normalization
        x = (self.a * mp.cos(theta_d)) / (self.a + 1)  # Normalize by n+1
        y = (self.a * mp.sin(theta_e)) / (self.a + 1)  # Normalize by n+1
        z = attrs['F'] / (E_SQUARED + attrs['F'])      # Self-normalizing ratio
        
        return (float(x), float(y), float(z))

    def get_4d_coordinates(self):
        attrs = self.attributes
        x, y, z = self.get_3d_coordinates()
        t = -self.c * (attrs['O'] / PHI)
        return (float(t), x, y, z)

    def get_5d_coordinates(self):
        attrs = self.attributes
        k_geo = self.get_curvature_geodesic_parameter()
        theta_d = PHI * ((attrs['D'] % PHI) / PHI) ** mp.mpf(k_geo)
        theta_e = PHI * ((attrs['E'] % PHI) / PHI) ** mp.mpf(k_geo)
        
        # Apply variance-minimizing normalization
        x = (self.a * mp.cos(theta_d)) / (self.a + 1)  # Normalize by n+1
        y = (self.a * mp.sin(theta_e)) / (self.a + 1)  # Normalize by n+1
        z = attrs['F'] / (E_SQUARED + attrs['F'])      # Self-normalizing ratio
        w = attrs['I'] / (1 + attrs['I'])              # Bounded normalization
        u = attrs['O'] / (1 + attrs['O'])              # Bounded normalization
        
        return (float(x), float(y), float(z), float(w), float(u))

    def get_5d_velocities(self, dt=1.0, c=299792458.0):
        """
        Computes 5D velocity components from coordinate derivatives with v_{5D}^2 = c^2 constraint.
        
        Uses finite differences to estimate velocity components v_x, v_y, v_z, v_t, v_w where:
        - v_i = (coord_i(n+1) - coord_i(n)) / dt for spatial dimensions
        - v_t represents temporal velocity component
        - v_w represents extra-dimensional velocity enforcing v_w > 0 for massive particles
        
        The constraint v_{5D}^2 = c^2 is enforced by normalizing the velocity vector.
        """
        # Get current and next coordinates
        current_coords = self.get_5d_coordinates()
        next_shift = self.unfold_next()
        next_coords = next_shift.get_5d_coordinates()
        
        # Compute velocity components via finite differences
        v_x = (next_coords[0] - current_coords[0]) / dt
        v_y = (next_coords[1] - current_coords[1]) / dt
        v_z = (next_coords[2] - current_coords[2]) / dt
        v_t = (next_coords[3] - current_coords[3]) / dt  # w-coordinate derivative as temporal velocity
        v_w_raw = (next_coords[4] - current_coords[4]) / dt  # u-coordinate derivative as extra-dimensional velocity
        
        # Compute 4D velocity magnitude
        v_4d_magnitude = np.sqrt(v_x**2 + v_y**2 + v_z**2 + v_t**2)
        
        # For massive particles, ensure v_w > 0 by deriving it from constraint
        # v_w = sqrt(c^2 - v_4d^2) ensures both constraint satisfaction and v_w > 0
        if v_4d_magnitude < c:
            v_w = np.sqrt(c**2 - v_4d_magnitude**2)
        else:
            # If 4D velocity exceeds c, normalize all components to maintain constraint
            normalization_factor = 0.95 * c / v_4d_magnitude  # Leave room for v_w > 0
            v_x *= normalization_factor
            v_y *= normalization_factor
            v_z *= normalization_factor
            v_t *= normalization_factor
            v_4d_magnitude *= normalization_factor
            v_w = np.sqrt(c**2 - v_4d_magnitude**2)
        
        return {
            'v_x': v_x,
            'v_y': v_y, 
            'v_z': v_z,
            'v_t': v_t,
            'v_w': v_w,
            'v_magnitude': c,
            'constraint_satisfied': True
        }

    def analyze_massive_particle_motion(self, c=299792458.0):
        """
        Analyzes massive particle motion along the w-dimension using curvature-based geodesics.
        
        For massive particles in 5D spacetime, the motion along the extra w-dimension is constrained by:
        1. v_{5D}^2 = c^2 (velocity constraint)
        2. v_w > 0 (massive particle requirement)
        3. Curvature-induced motion via κ(n) = d(n) * ln(n+1) / e^2
        
        Returns analysis of w-dimension motion characteristics and geodesic properties.
        """
        # Get velocity components
        velocities = self.get_5d_velocities(c=c)
        
        # Compute discrete curvature
        n = int(self.a)
        d_n = len(divisors(n))
        kappa = d_n * mp.log(n + 1) / E_SQUARED
        
        # Analyze w-motion characteristics
        v_w = velocities['v_w']
        is_massive = v_w > 0
        
        # Connect to curvature: lower curvature (primes) should have different w-motion
        is_prime = isprime(n)
        
        # Compute Kaluza-Klein charge-induced motion component
        from .axioms import curvature_induced_w_motion
        curvature_w_component = curvature_induced_w_motion(n, d_n, c)
        
        return {
            'n': n,
            'v_w': v_w,
            'is_massive_particle': is_massive,
            'is_prime': is_prime,
            'discrete_curvature': float(kappa),
            'curvature_induced_w_velocity': curvature_w_component,
            'w_motion_type': 'charge_induced' if is_prime else 'curvature_enhanced',
            'geodesic_classification': 'minimal_curvature' if is_prime else 'standard_curvature'
        }

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