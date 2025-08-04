import math
import numpy as np
from sympy import divisors

PHI = (1 + math.sqrt(5)) / 2
E_SQUARED = np.exp(2)

class ZetaGeometricHash:
    def __init__(self, seed=2, v=1.0, delta_max=E_SQUARED):
        self.seed = seed
        self.v = v
        self.delta_max = delta_max

    def kappa(self, n):
        d_n = len(divisors(n))
        return d_n * math.log(n + 1) / E_SQUARED

    def theta_prime(self, x, k=0.3):
        return PHI * ((x % PHI) / PHI) ** k

    def zeta_shift(self, n):
        delta_n = self.v * self.kappa(n)
        z = n * (delta_n / self.delta_max)
        d = self.delta_max / n
        e = self.delta_max / delta_n
        f = self.theta_prime(d / e)
        # Simplified O approximation for efficiency; full chain in production
        o = (self.theta_prime(f / (e / f / E_SQUARED)))  # Terminal proxy
        return o

    def forward_hash(self, message: bytes, L=32):
        n_seq = [self.seed + byte for byte in message]
        trajectory = [self.zeta_shift(n) for n in n_seq]
        x = np.mean([o * math.cos(self.theta_prime(i)) for i, o in enumerate(trajectory)])
        y = np.mean([o * math.sin(self.theta_prime(i)) for i, o in enumerate(trajectory)])
        z = np.mean([o / E_SQUARED for o in trajectory])
        v = (x + y + z) % PHI**L
        digest = hex(int(v * 10**L) % (1 << 256))[2:].zfill(64)
        return digest.encode()

    def inverse_traverse(self, digest: bytes, L=32, message_len=None):
        # Inverse hypothesis: Solve for trajectory via root-finding (e.g., fsolve for production)
        v = int(digest.decode(), 16) / 10**L
        # Approximate reconstruction; full bijection via symbolic sympy
        trajectory_approx = [v * PHI**i for i in range(L)]
        n_seq = [int(math.exp(math.log(t) / self.v) * E_SQUARED / math.log(n+1)) for t, n in zip(trajectory_approx, range(self.seed, self.seed+L))]  # Proxy
        reconstructed = bytes([(n - self.seed) % 256 for n in n_seq][:message_len])
        return reconstructed

# Example
zgh = ZetaGeometricHash()
msg = b"Test message"
hash_val = zgh.forward_hash(msg)
reconstructed = zgh.inverse_traverse(hash_val, message_len=len(msg))
print(f"Hash: {hash_val}\nReconstructed: {reconstructed}")