import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpmath import zetazero
from sympy import divisor_count, isprime
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF

# 1. Load the raw zeta shifts (n, a, b, c, z, D…O)
df = pd.read_csv("zeta_shifts_1_to_6000.csv")

# 2. Recompute κ(n) = d(n)·ln(n+1)/e²
df["kappa"] = df["n"].apply(divisor_count) * np.log(df["n"] + 1) / np.exp(2)

# 3. Compute prime gaps
df["is_prime"] = df["n"].apply(isprime)
primes = df.loc[df["is_prime"], "n"].values
gaps = np.diff(primes)
gap_map = {p: g for p, g in zip(primes[:-1], gaps)}
df["prime_gap"] = df["n"].map(gap_map).fillna(0).astype(int)

# 4. Fit GMM on (z, κ, prime_gap) and pull out cluster 2
features = df[["z", "kappa", "prime_gap"]].fillna(0)
gmm = GaussianMixture(n_components=3, random_state=42).fit(features)
df["cluster"] = gmm.predict(features)
cluster2 = df[df["cluster"] == 2].copy()
n_vals = np.sort(cluster2["n"].values)

print(f"Cluster 2 size: {len(n_vals)}")
print(f"Cluster 2 median Z: {cluster2['z'].median():.1f}")

# 5. Fetch & scale first 20 Riemann zeros to [min(n), max(n)]
zeros = np.array([float(zetazero(k)) for k in range(1, 21)])
min_n, max_n = n_vals.min(), n_vals.max()
scale = (max_n - min_n) / (zeros.max() - zeros.min())
zeros_scaled = zeros * scale + (min_n - zeros.min() * scale)

# 6. Compute nearest‐zero distance for each n in cluster2
dists = np.min(np.abs(n_vals[:, None] - zeros_scaled[None, :]), axis=1)
print(f"Nearest‐zero distances: min={dists.min():.4f}, mean={dists.mean():.4f}, max={dists.max():.4f}")

# 7. Spacing distribution vs. GUE Wigner surmise
spacings = np.diff(n_vals)
sp_norm = spacings / spacings.mean()

# simulate GUE spacings via inverse‐CDF sampling
u = np.random.rand(200_000)
s_gue = np.sqrt(-4/np.pi * np.log(1 - u))

stat, pval = ks_2samp(sp_norm, s_gue)
print(f"KS statistic={stat:.4f}, p‐value={pval:.2e}")

# 8. Plot ECDF comparison
ecdf_emp = ECDF(sp_norm)
ecdf_gue = ECDF(s_gue)

plt.step(ecdf_emp.x, ecdf_emp.y, label="Cluster 2")
plt.step(ecdf_gue.x, ecdf_gue.y, label="GUE Wigner", alpha=0.7)
plt.xlabel("s/⟨s⟩")
plt.ylabel("CDF")
plt.title("Cluster 2 Spacings vs. GUE Surmise")
plt.legend()
plt.show()
