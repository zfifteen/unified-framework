def golden_curve(n):
    # This function computes the golden curve for a given n
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    return [phi ** i for i in range(n)]

# Example usage
if __name__ == '__main__':
    n = 10
    print(golden_curve(n))