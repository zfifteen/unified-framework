import domain
import csv

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

# Loop for the first 50 numbers
for n in range(1, 51):
    
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

    # Store the results
    results.append([n, z_value])

    # Print the results
    print(f"Number n: {n}")
    print(f"Z(n) value: {z_value}")

# Export results to CSV file
with open('zeta_shift_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    csv_writer.writerow(['Number (n)', 'Z(n) Value'])

    # Write data rows
    csv_writer.writerows(results)

print("\nResults have been exported to 'zeta_shift_results.csv'.")

# Exit
exit()