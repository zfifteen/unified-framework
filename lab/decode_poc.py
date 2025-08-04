import math
import hashlib
import collections

def divisor_count(n):
    n = int(n)
    if n <= 0:
        return 0
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1 if i * i == n else 2
    return count

class DiscreteZetaShift:
    def __init__(self, n, v=1.0, delta_max=math.e**2):
        self.a = n
        kappa = divisor_count(self.a) * math.log(self.a + 1) / math.e**2
        self.b = v * kappa
        self.c = delta_max
        self.z = self.a * (self.b / self.c)
        self.D = self.c / self.a if self.a != 0 else float('inf')
        self.E = self.c / self.b if self.b != 0 else float('inf')
        self.phi = (1 + math.sqrt(5)) / 2
        self.k = 0.3
        self.F = self.theta_prime(self.D / self.E) if math.isfinite(self.D / self.E) else 0
        self.G = (self.E / self.F) / math.e**2 if self.F != 0 else 0
        self.H = self.F / self.G if self.G != 0 else 0
        self.I = self.theta_prime(self.G / self.H) if math.isfinite(self.G / self.H) else 0
        self.J = self.H / self.I if self.I != 0 else 0
        self.K = (self.I / self.J) / math.e**2 if self.J != 0 else 0
        self.L = self.J / self.K if self.K != 0 else 0
        self.M = self.theta_prime(self.K / self.L) if math.isfinite(self.K / self.L) else 0
        self.N = self.L / self.M if self.M != 0 else 0
        self.O = self.M / self.N if self.N != 0 else 0
        self.vortex = collections.deque(maxlen=int(self.phi * math.pi))

    def theta_prime(self, x):
        mod_part = (x % self.phi) / self.phi
        return self.phi * (mod_part ** self.k)

    def unfold_next(self):
        next_n = self.a + 1
        next_shift = DiscreteZetaShift(next_n)
        self.vortex.append(next_shift.O)
        return next_shift.O

    def generate_key(self, N, seed_n):
        self.a = seed_n
        key_str = str(self.O)
        for _ in range(N):
            o = self.unfold_next()
            key_str += str(o)
        hash_obj = hashlib.sha256(key_str.encode())
        return hash_obj.digest()

def repeating_xor_decrypt(ciphertext, key):
    key = key * (len(ciphertext) // len(key) + 1)
    plaintext = bytes(c ^ k for c, k in zip(ciphertext, key))
    return plaintext

def z_cipher_decode(ciphertext, seed_range=range(2, 100), N=5, num_candidates=20, key_len=1, language_checker=lambda x: len(x) > 0):
    candidates = []
    for seed in seed_range:
        zeta = DiscreteZetaShift(seed)
        if zeta.O < 5.0:
            key = zeta.generate_key(N, seed)[:key_len]
            candidates.append(key)
            if len(candidates) >= num_candidates:
                break
    for key in candidates:
        plaintext = repeating_xor_decrypt(ciphertext, key)
        if language_checker(plaintext):
            return plaintext, key
    return None, None