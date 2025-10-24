import unittest
import numpy as np
from src.ldpc_code_construction import generate_regular_ldpc
from src.belief_propagation import bp_decode

class TestLDPC(unittest.TestCase):
    
    def test_ldpc_construction(self):
        n = 100
        dv = 3
        dc = 6
        H = generate_regular_ldpc(n, dv, dc)
        self.assertEqual(H.shape, (n*dv//dc, n))
        self.assertTrue(np.all(H.sum(axis=0) == dv))
        self.assertTrue(np.all(H.sum(axis=1) == dc))
    
    def test_bp_decode(self):
        # Test with a simple parity-check matrix (3,6) code, but with a small example
        H = np.array([[1, 1, 0, 1, 0, 0],
                      [1, 0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0, 1]], dtype=int)
        # All-zero codeword
        received_llr = np.array([1, 1, 1, 1, 1, 1])  # Strong positive LLRs (should decode to 0)
        decoded_bits, success = bp_decode(H, received_llr)
        self.assertTrue(success)
        self.assertTrue(np.all(decoded_bits == 0))

if __name__ == '__main__':
    unittest.main()
