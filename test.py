import core.axioms
from core.domain import DiscreteZetaShift
import csv

with open("z_shift_embeddings.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Collect all possible attribute names from the first 100 DiscreteZetaShift objects
    attribute_names = set()
    z_instances = [DiscreteZetaShift(i + 1) for i in range(100)]
    for z in z_instances:
        if hasattr(z, "__dict__"):
            attribute_names.update(z.__dict__.keys())
    attribute_names = sorted(attribute_names)  # Sorted for consistent column order

    # Prepare header: i, z_value, z_str, then all attribute columns
    header = ["i", "z_value", "z_str"] + attribute_names
    writer.writerow(header)

    for i, z in enumerate(z_instances):
        z_value = getattr(z, "value", None) if hasattr(z, "value") else None
        z_str = str(z)
        # Extract attribute values matching header order
        attrs = []
        if hasattr(z, "__dict__"):
            z_dict = z.__dict__
            attrs = [z_dict.get(attr, None) for attr in attribute_names]
        else:
            attrs = [None for _ in attribute_names]

        writer.writerow([i, z_value, z_str] + attrs)