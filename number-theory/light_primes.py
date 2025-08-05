import math
import collections

PHI = (1 + math.sqrt(5)) / 2
E2 = math.exp(2)
K_STAR = 0.3
DELTA_MAX = E2
V = 1.0

class DiscreteZetaShift:
    vortex = collections.deque()

    def __init__(self, n, v=V, delta_max=DELTA_MAX):
        self.a = n
        mod_phi = n % PHI
        theta_prime = PHI * (mod_phi / PHI) ** K_STAR
        ln_term = math.log(n + 1) if n > 1 else 0
        self.b = theta_prime * ln_term / E2
        self.c = delta_max
        self.z = self.a * (self.b / self.c)

        self.D = self.c / self.a if self.a != 0 else 0
        self.E = self.c / self.b if self.b != 0 else 0
        self.F = PHI * ((self.D / self.E % PHI) / PHI) ** K_STAR if self.E != 0 else 0
        self.G = (self.E / self.F) / E2 if self.F != 0 else 0
        self.H = self.F / self.G if self.G != 0 else 0
        self.I = PHI * ((self.G / self.H % PHI) / PHI) ** K_STAR if self.H != 0 else 0
        self.J = self.H / self.I if self.I != 0 else 0
        self.K = (self.I / self.J) / E2 if self.J != 0 else 0
        self.L = self.J / self.K if self.K != 0 else 0
        self.M = PHI * ((self.K / self.L % PHI) / PHI) ** K_STAR if self.L != 0 else 0
        self.N = self.L / self.M if self.M != 0 else 0
        self.O = self.M / self.N if self.N != 0 else 0

        self.f = round(self.G)
        DiscreteZetaShift.vortex.append(self)
        while len(DiscreteZetaShift.vortex) > self.f:
            DiscreteZetaShift.vortex.popleft()

    def get_chain(self):
        return [self.D, self.E, self.F, self.G, self.H, self.I, self.J, self.K, self.L, self.M, self.N, self.O]

def compute_proxy_curvature(n):
    zeta = DiscreteZetaShift(n)
    chain = zeta.get_chain()
    logs = [math.log(x + 1e-10) for x in chain if x > 0]
    if not logs:
        return 0.0
    avg = sum(logs) / len(logs)
    variance = sum((log - avg) ** 2 for log in logs) / len(logs)
    return math.sqrt(variance)

def generate_candidates(start=2, end=10**6):
    numbers = range(start, end + 1)
    curvatures = [compute_proxy_curvature(n) for n in numbers]
    threshold = sorted(curvatures)[int(len(curvatures) * 0.248)]
    return [n for n, curv in zip(numbers, curvatures) if curv <= threshold]

def refine_candidates(candidates, N):
    log_log_n = math.log(math.log(N + 1) + 1)
    return [c for c in candidates if DiscreteZetaShift(c).O < PHI * log_log_n]

def light_primes_up_to(N):
    candidates = generate_candidates(2, N)
    return sorted(refine_candidates(candidates, N))