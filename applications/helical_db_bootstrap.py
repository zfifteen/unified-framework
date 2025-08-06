import sqlite3
import numpy as np
import time

def main():
    limit = 50_000_000
    phi = (1 + np.sqrt(5)) / 2
    e = np.exp(1)
    k = 0.3
    batch_size = 1_000_000  # Batch size for computation and insertion

    start_time = time.time()
    last_print = start_time

    # Compute divisor counts using sieve (O(limit log limit))
    print("Starting divisor sieve...")
    divisors = np.zeros(limit + 1, dtype=np.int32)
    for i in range(1, limit + 1):
        divisors[i::i] += 1
        if i % 100_000 == 0:
            current_time = time.time()
            if current_time - last_print >= 30:
                print(f"Sieve progress: {i}/{limit} ({i / limit * 100:.2f}%), time elapsed: {current_time - start_time:.2f} sec")
                last_print = current_time
    print("Sieve completed.")

    # Connect to database and create table
    conn = sqlite3.connect('helical_embeddings.db')
    conn.execute('''
                 CREATE TABLE IF NOT EXISTS embeddings (
                                                           n INTEGER PRIMARY KEY,
                                                           x REAL,
                                                           y REAL,
                                                           z REAL
                 )
                 ''')
    conn.commit()

    # Process in batches
    for start in range(1, limit + 1, batch_size):
        end = min(start + batch_size - 1, limit)
        n_arr = np.arange(start, end + 1, dtype=np.float64)  # Ensure float for computations

        # Compute curvature components
        ln = np.log(n_arr + 1)
        kappa = divisors[start:end + 1] * ln / e**2

        # Prime curvature transformation for angular positioning
        frac = np.mod(n_arr, phi) / phi
        curved_frac = np.power(frac, k)
        theta = 2 * np.pi * curved_frac

        # Helical embedding coordinates
        r = kappa
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = n_arr / np.exp(kappa)  # Z-normalized axial coordinate

        # Check for invalid values (NaN or inf)
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
            raise ValueError(f"Invalid values (NaN or inf) detected in batch {start}-{end}")

        # Prepare data with built-in types to prevent datatype mismatch
        data = [(int(n), float(xx), float(yy), float(zz)) for n, xx, yy, zz in zip(n_arr, x, y, z)]

        # Batch insert
        conn.executemany('INSERT INTO embeddings VALUES (?, ?, ?, ?)', data)
        conn.commit()

        # Progress update
        current_time = time.time()
        if current_time - last_print >= 30:
            print(f"Inserted batch up to {end}, progress: {end / limit * 100:.2f}%, time elapsed: {current_time - start_time:.2f} sec")
            last_print = current_time

    conn.close()
    total_time = time.time() - start_time
    print(f"Completed all operations. Total time: {total_time:.2f} sec")

if __name__ == '__main__':
    main()