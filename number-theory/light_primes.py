import math
import cmath
import collections
import random
import logging

PHI = (1 + math.sqrt(5)) / 2
E2 = math.exp(2)
K_STAR = 0.3
DELTA_MAX = E2
V = 1.0
UNFOLD_ITERS = 4  # Base; cap at 3 for small N
BOOTSTRAP_SAMPLES = 100  # For adaptive quantile
SAMPLE_FRACTION = 0.1  # For entropy median approximation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('light_primes.log'), logging.StreamHandler()])

class DiscreteZetaShift:
    vortex = collections.deque()

    def __init__(self, n, v=V, delta_max=DELTA_MAX, max_n=10**6):
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

        # Scale-dependent vortex bounding
        log_log_max = math.log(math.log(max_n + 1) + 1) if max_n >= 100 else 0
        self.f = 3 if max_n < 100 else round(self.G) + log_log_max
        DiscreteZetaShift.vortex.append(self)
        while len(DiscreteZetaShift.vortex) > self.f:
            DiscreteZetaShift.vortex.popleft()

    def get_chain(self):
        return [self.D, self.E, self.F, self.G, self.H, self.I, self.J, self.K, self.L, self.M, self.N, self.O]

    def unfold_next(self):
        successor = DiscreteZetaShift(self.a + 1, v=self.b, delta_max=self.c, max_n=self.a + 1)  # max_n propagates approximately
        DiscreteZetaShift.vortex.append(successor)
        while len(DiscreteZetaShift.vortex) > successor.f:
            DiscreteZetaShift.vortex.popleft()
        return successor

    def get_helical_coords(self):
        theta_d = PHI * ((self.D % PHI) / PHI) ** K_STAR
        theta_e = PHI * ((self.E % PHI) / PHI) ** K_STAR
        x = self.a * math.cos(theta_d)
        y = self.a * math.sin(theta_e)
        z = self.F / E2
        return x, y, z

def compute_proxy_curvature(n, max_n):
    unfold_iters = 3 if max_n < 1000 else UNFOLD_ITERS  # Cap at lower N
    zeta = DiscreteZetaShift(n, max_n=max_n)
    o_values = [zeta.O]
    for _ in range(unfold_iters - 1):
        zeta = zeta.unfold_next()
        o_values.append(zeta.O)
    logs = [math.log(o + 1e-10) for o in o_values if o > 0]
    if not logs:
        return 0.0
    avg = sum(logs) / len(logs)
    variance = sum((log - avg) ** 2 for log in logs) / len(logs)
    logging.info(f"Curvature for n={n}: variance={variance:.4f}")
    return math.sqrt(variance)

def generate_candidates(start=2, end=10**6):
    numbers = range(start, end + 1)
    curvatures = [compute_proxy_curvature(n, max_n=end) for n in numbers]
    logging.info(f"Curvatures computed: min={min(curvatures):.4f}, max={max(curvatures):.4f}, mean={sum(curvatures)/len(curvatures):.4f}")

    # Adaptive quantile via bootstrap, tune target_elim for larger N
    target_elim = 0.7 if end > 10**5 else 0.752
    quantiles = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = random.choices(curvatures, k=len(curvatures))
        sorted_sample = sorted(sample)
        quantiles.append(sorted_sample[int(len(sorted_sample) * (1 - target_elim))])
    threshold = sum(quantiles) / len(quantiles)
    logging.info(f"Curvature threshold: {threshold:.4f}")

    candidates = [n for n, curv in zip(numbers, curvatures) if curv <= threshold]
    logging.info(f"Candidates after curvature filter: {len(candidates)}")
    return candidates

def refine_candidates(candidates, N):
    z_threshold = PHI / 1.5
    logging.info(f"Helical z threshold: {z_threshold:.4f}")
    # Sample for O percentile
    sample_size = max(10, int(N * SAMPLE_FRACTION))
    sample_ns = random.choices(range(2, N+1), k=sample_size)
    sample_O = [DiscreteZetaShift(i, max_n=N).O for i in sample_ns]
    percentile_60_O = sorted(sample_O)[int(len(sample_O) * 0.6)]
    min_O = min(sample_O)
    max_O = max(sample_O)
    mean_O = sum(sample_O) / sample_size
    logging.info(f"60th percentile O (sampled): {percentile_60_O:.4f}, min={min_O:.4f}, max={max_O:.4f}, mean={mean_O:.4f}")

    refined = []
    filter_counts = {"high_z": 0, "high_O": 0}
    for c in candidates:
        zeta = DiscreteZetaShift(c, max_n=N)
        # Helical thresholding
        _, _, z = zeta.get_helical_coords()
        logging.info(f"Helical z for n={c}: {z:.4f}")
        if z >= z_threshold:
            filter_counts["high_z"] += 1
            logging.info(f"Filtered n={c} (high z: {z:.4f} >= {z_threshold:.4f})")
            continue
        # O filter (percentile-based)
        logging.info(f"O for n={c}: {zeta.O:.4f}")
        if zeta.O > percentile_60_O:
            filter_counts["high_O"] += 1
            logging.info(f"Filtered n={c} (high O: {zeta.O:.4f} > {percentile_60_O:.4f})")
            continue
        refined.append(c)
        logging.info(f"Retained n={c}")
    logging.info(f"Refined candidates: {len(refined)}")
    logging.info(f"Filtration counts: high_z={filter_counts['high_z']}, high_O={filter_counts['high_O']}")
    return refined

def light_primes_in_range(start=2, end=1000):
    candidates = generate_candidates(start, end)
    return sorted(refine_candidates(candidates, end))