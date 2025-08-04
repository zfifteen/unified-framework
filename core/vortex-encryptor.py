import math
import numpy as np
from sympy import divisors
import collections
import hashlib
import secrets  # For secure randomness if needed

from core.domain import UniversalZetaShift
from domain import UniversalZetaShift

PHI = (1 + math.sqrt(5)) / 2
E_SQUARED = np.exp(2)

class VortexEncryptor(UniversalZetaShift):
    vortex = collections.deque()  # Ephemeral vortex for short-lived keys

    def __init__(self, n, v=1.0, delta_max=E_SQUARED):
        if n < 2:
            raise ValueError("n must be at least 2")
        d_n = len(divisors(n))
        kappa = d_n * math.log(n + 1) / E_SQUARED
        delta_n = v * kappa
        super().__init__(a=n, b=delta_n, c=delta_max)
        self.v = v
        self.f = round(self.getG())  # Bound vortex to ~π
        self.vortex.append(self)
        while len(self.vortex) > self.f:
            self.vortex.popleft()

    def unfold_next(self):
        successor = VortexEncryptor(self.a + 1, v=self.v, delta_max=self.c)
        self.vortex.append(successor)
        while len(self.vortex) > successor.f:
            self.vortex.popleft()
        return successor

    @classmethod
    def generate_trajectory(cls, N=3, seed_n=2, v=1.0, delta_max=E_SQUARED):
        """Generate ephemeral O trajectory for key stream"""
        if N < 1:
            raise ValueError("N must be at least 1")
        zeta = cls(seed_n, v, delta_max)
        trajectory_o = [zeta.getO()]
        for _ in range(1, N):
            zeta = zeta.unfold_next()
            trajectory_o.append(zeta.getO())
        # Clear vortex for ephemerality
        cls.vortex.clear()
        return trajectory_o

    @classmethod
    def trajectory_to_keystream(cls, trajectory, key_length):
        """Convert O trajectory to byte keystream using geometric hashing"""
        # Concatenate formatted O values
        hash_input = ''.join(f"{o:.10f}" for o in trajectory).encode()
        # Use SHA-256 for initial hash, then expand to desired length via HKDF-like derivation
        base_hash = hashlib.sha256(hash_input).digest()
        keystream = b''
        while len(keystream) < key_length:
            keystream += hashlib.sha256(base_hash + len(keystream).to_bytes(4, 'big')).digest()
        return keystream[:key_length]

    @classmethod
    def encrypt(cls, plaintext: bytes, N=3, seed_n=2, v=1.0, delta_max=E_SQUARED) -> bytes:
        """Encrypt using ephemeral vortex trajectory as keystream (XOR for reversibility)"""
        trajectory = cls.generate_trajectory(N, seed_n, v, delta_max)
        keystream = cls.trajectory_to_keystream(trajectory, len(plaintext))
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream))
        return ciphertext

    @classmethod
    def decrypt(cls, ciphertext: bytes, N=3, seed_n=2, v=1.0, delta_max=E_SQUARED) -> bytes:
        """Decrypt identically to encrypt due to XOR symmetry"""
        return cls.encrypt(ciphertext, N, seed_n, v, delta_max)  # XOR is involution

# Use Cases
def use_case_1_message_encryption():
    """Use Case 1: Secure message encryption"""
    message = b"Secret message"
    ciphertext = VortexEncryptor.encrypt(message, N=5, seed_n=3)
    decrypted = VortexEncryptor.decrypt(ciphertext, N=5, seed_n=3)
    assert decrypted == message, "Decryption failed"
    print("Use Case 1: Message encrypted and decrypted successfully.")

def use_case_2_file_encryption():
    """Use Case 2: File content encryption"""
    file_content = b"File data to protect"
    ciphertext = VortexEncryptor.encrypt(file_content, N=4, seed_n=5, v=0.5)
    decrypted = VortexEncryptor.decrypt(ciphertext, N=4, seed_n=5, v=0.5)
    assert decrypted == file_content, "File decryption failed"
    print("Use Case 2: File encrypted and decrypted successfully.")

def use_case_3_session_key():
    """Use Case 3: Ephemeral session key for communication"""
    session_data = secrets.token_bytes(16)  # Simulate session data
    ciphertext = VortexEncryptor.encrypt(session_data, N=3, seed_n=7)
    decrypted = VortexEncryptor.decrypt(ciphertext, N=3, seed_n=7)
    assert decrypted == session_data, "Session decryption failed"
    print("Use Case 3: Session key encrypted and decrypted successfully.")

# Unit Tests
def test_reversibility():
    """Unit Test 1: Verify encryption-decryption reversibility"""
    data = b"Test data"
    ciphertext = VortexEncryptor.encrypt(data, N=3, seed_n=2)
    decrypted = VortexEncryptor.decrypt(ciphertext, N=3, seed_n=2)
    assert decrypted == data, "Reversibility test failed"

def test_seed_consistency():
    """Unit Test 2: Verify consistency with same seed/parameters"""
    data = b"Consistent test"
    cipher1 = VortexEncryptor.encrypt(data, N=4, seed_n=11, v=1.5)
    cipher2 = VortexEncryptor.encrypt(data, N=4, seed_n=11, v=1.5)
    assert cipher1 == cipher2, "Seed consistency test failed"

def test_error_handling():
    """Unit Test 3: Verify error handling for invalid parameters"""
    try:
        VortexEncryptor.encrypt(b"Data", N=0)
        assert False, "Should raise error for N<1"
    except ValueError:
        pass  # Expected

# Run demonstrations
if __name__ == "__main__":
    use_case_1_message_encryption()
    use_case_2_file_encryption()
    use_case_3_session_key()
    test_reversibility()
    test_seed_consistency()
    test_error_handling()
    print("All tests passed.")