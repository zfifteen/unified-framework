import numpy as np
from scipy.fft import fft
from scipy.stats import entropy
from sympy import primerange, divisors, isprime
from math import log, exp, sqrt, pi, sin, cos
import pandas as pd  # Added for CSV loading
from core import axioms, domain
from core.domain import DiscreteZetaShift

PHI = (1 + sqrt(5)) / 2  # Golden ratio
E2 = exp(2)  # e^2 for curvature normalization

def divisor_count(n):
    """Compute number of divisors d(n)."""
    return len(divisors(int(n)))

def curvature(n):
    """Frame-normalized curvature κ(n) = d(n) · ln(n+1) / e²."""
    if n < 1:
        return 0
    return divisor_count(n) * log(n + 1) / E2

def z_normalize(n, v=1.0):
    """Z-normalization Z(n) = n / exp(v · κ(n)), with velocity v."""
    return n / exp(v * curvature(n))

def golden_transform(n, k=0.3):
    """Prime curvature transformation θ'(n,k) = φ · ((n mod φ)/φ)^k."""
    mod_phi = n % PHI
    return PHI * (mod_phi / PHI) ** k

def load_zeta_csv(csv_path):
    """Load zeta shift embeddings from CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    df['is_prime'] = df['index'].apply(isprime)  # Add prime flag using sympy
    return df

def encode_waveform(sequence, window_size=1024, use_z=True, v=1.0):
    """Encode sequence as complex waveform Ψ_n = w_n · e^{2πi s_n}."""
    waveforms = []
    for i in range(0, len(sequence) - window_size + 1, window_size // 2):  # Overlapping windows
        window = sequence[i:i + window_size]
        weights = [z_normalize(val, v) if use_z else val for val in window]
        spacings = np.cumsum(weights) / np.sum(weights)  # Normalized cumulative spacing
        waveform = np.array(weights) * np.exp(2j * pi * spacings)
        waveforms.append(waveform)
    return waveforms

def compute_spectral_features(waveform):
    """Compute FFT-based features: dominant freq shift, peaks, entropy."""
    if len(waveform) < 2:
        # Not enough data for spectral analysis; return safe defaults.
        return 0.0, 0, 0.0

    spectrum = np.abs(fft(waveform))
    spectrum = spectrum / np.sum(spectrum)  # Normalize to probability distribution
    freqs = np.fft.fftfreq(len(waveform))

    # Dominant frequency shift Δf_1 (shift from zero-freq)
    dom_idx = np.argmax(spectrum[1:]) + 1  # Exclude DC component
    delta_f1 = abs(freqs[dom_idx])

    # ΔPeaks: Number of side-lobes (peaks above threshold)
    threshold = 0.1 * np.max(spectrum)
    delta_peaks = np.sum(spectrum > threshold) - 1  # Exclude main peak

    # ΔEntropy: Spectral entropy H = -∑ p_i log p_i
    delta_entropy = entropy(spectrum)

    return delta_f1, delta_peaks, delta_entropy

def disruption_score(waveforms, ref_waveforms=None, use_z=True, v=1.0):
    """Composite Wave-CRISPR score for disruption quantification."""
    scores = []
    if ref_waveforms:
        n = min(len(waveforms), len(ref_waveforms))
    else:
        n = len(waveforms)
    for i in range(n):
        wf = waveforms[i]
        delta_f1, delta_peaks, delta_entropy = compute_spectral_features(wf)
        z_n = z_normalize(i + 1, v) if use_z else 1.0  # Position-based Z
        score = z_n * abs(delta_f1) + delta_peaks + delta_entropy

        if ref_waveforms:
            ref_delta_f1, ref_delta_peaks, ref_delta_entropy = compute_spectral_features(ref_waveforms[i])
            score = abs(score - (z_n * abs(ref_delta_f1) + ref_delta_peaks + ref_delta_entropy))

        scores.append(score)

    if not scores:
        # Handle the empty case gracefully.
        return 0.0

    return np.mean(scores)

# Example usage with zeta CSV integration
# todo: instead of loading from file, create DiscreetZetShift = DiscreteZetaShift(n) and use the attributes so we can create as many as we want

if __name__ == "__main__":
    csv_path = "../../z_shift_embeddings_descriptive.csv"  # Replace with actual path
    df = load_zeta_csv(csv_path)

    # Full sequence (e.g., rate_b)
    full_seq = df['rate_b'].values
    full_waveforms = encode_waveform(full_seq)
    full_score = disruption_score(full_waveforms)
    print(f"Full sequence disruption score: {full_score}")

    # Prime subsequence
    prime_df = df[df['is_prime']]
    prime_seq = prime_df['rate_b'].values
    prime_waveforms = encode_waveform(prime_seq)
    prime_score = disruption_score(prime_waveforms)
    print(f"Prime subsequence disruption score: {prime_score}")

    # Composite subsequence
    composite_df = df[~df['is_prime']]
    composite_seq = composite_df['rate_b'].values
    composite_waveforms = encode_waveform(composite_seq)
    composite_score = disruption_score(composite_waveforms)
    print(f"Composite subsequence disruption score: {composite_score}")