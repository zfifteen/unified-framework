import domain
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
for the first 50 numbers
- instantiate a PhysicalZetaShift z(n) for that number
- print out the number n
- print out the z(n) value
- export results to a CSV file
"""
# Importing the required class
from domain import PhysicalZetaShift

# --- Helper function to check for primality ---
def is_prime(num):
    """Checks if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Prepare a list to store results for CSV export
results = []

# --- Variables to track prime gaps ---
largest_gap = 0
last_prime = 2

# Loop for the first 6000 numbers
for n in range(1, 6001):
    
    # Check if n is prime and update the largest gap
    if is_prime(n):
        gap = n - last_prime
        if gap > largest_gap:
            largest_gap = gap
        last_prime = n
    
    # "v" is the largest encountered gap size up to n
    v = largest_gap
    
    zeta_shift = PhysicalZetaShift(t=n, v=v)  # Instantiate PhysicalZetaShift
    z_value = zeta_shift.compute_z()

    # Store the results, including v
    results.append([n, z_value, v])

    # Print the results
    print(f"Number n: {n}")
    print(f"Z(n) value: {z_value}")
    print(f"Largest gap 'v': {v}")

# Export results to CSV file
csv_filename = 'zeta_shift_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    csv_writer.writerow(['Number (n)', 'Z(n) Value', 'Largest Gap (v)'])

    # Write data rows
    csv_writer.writerows(results)

print(f"\nResults have been exported to '{csv_filename}'.")

# --- Read from CSV and create a 3D plot ---
n_vals, z_vals, v_vals = [], [], []
with open(csv_filename, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        n_vals.append(float(row[0]))
        z_vals.append(float(row[1]))
        v_vals.append(float(row[2]))

# Create the 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(n_vals, v_vals, z_vals, c=z_vals, cmap='viridis', marker='o', alpha=0.6)

# Set labels and title
ax.set_xlabel('Number (n)')
ax.set_ylabel('Largest Gap (v)')
ax.set_zlabel('Z(n) Value')
ax.set_title('3D Plot of Zeta Shift vs. n and Largest Prime Gap')

# Save the plot to a file
plot_filename = 'zeta_shift_3d_plot.png'
plt.savefig(plot_filename)

print(f"3D plot has been saved to '{plot_filename}'.")


# Exit
exit()