import core.axioms
from core.domain import DiscreteZetaShift
import csv
import numpy as np  # For potential spectral metrics

# Descriptive mapping
attr_map = {
    'a': 'frame_dependent_measure_a',
    'b': 'rate_b',
    'c': 'universal_invariant_c',
    'z': 'computed_z',
    'D': 'zeta_shift_level_1_D',
    'E': 'zeta_shift_level_2_E',
    'F': 'zeta_shift_level_3_F',
    'G': 'zeta_shift_level_4_G',
    'H': 'zeta_shift_level_5_H',
    'I': 'zeta_shift_level_6_I',
    'J': 'zeta_shift_level_7_J',
    'K': 'zeta_shift_level_8_K',
    'L': 'zeta_shift_level_9_L',
    'M': 'zeta_shift_level_10_M',
    'N': 'zeta_shift_level_11_N',
    'O': 'zeta_shift_level_12_O'
}

with open("z_shift_embeddings_descriptive.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Fixed header with descriptive names
    header = ['index'] + [attr_map[attr] for attr in sorted(attr_map.keys())]
    writer.writerow(header)

    # Generate for n=1 to 1000 for robustness
    for i in range(1, 5001):
        z = DiscreteZetaShift(i)
        row = [i]
        z_dict = z.__dict__
        for attr in sorted(attr_map.keys()):
            row.append(z_dict.get(attr, np.nan))  # NaN for missing, though complete
        writer.writerow(row)

    # Optional: Append spectral entropy column (wave-CRISPR adaptation)
    # Load shifts, compute H = -sum(p log p) over normalized [D:O] per row