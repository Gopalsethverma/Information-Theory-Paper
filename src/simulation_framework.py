import numpy as np
import matplotlib.pyplot as plt
from ldpc_code_construction import generate_regular_ldpc
from belief_propagation import bp_decode

def simulate_ldpc(n, dv, dc, snr_db_list, max_iter=50, num_frames=1000):
    """
    Run Monte Carlo simulation for LDPC codes over AWGN channel.
    Args:
        n: block length
        dv: variable node degree
        dc: check node degree
        snr_db_list: list of SNR values in dB
        max_iter: maximum BP iterations
        num_frames: number of frames to simulate per SNR point
    Returns:
        ber_list: Bit Error Rate for each SNR
        fer_list: Frame Error Rate for each SNR
    """
    m = n * dv // dc
    H = generate_regular_ldpc(n, dv, dc)
    
    ber_list = []
    fer_list = []
    
    for snr_db in snr_db_list:
        snr_linear = 10**(snr_db/10)
        # For BPSK, the noise variance per dimension is N0/2 = 1/(2*SNR) since signal power is 1.
        noise_var = 1.0 / (2 * snr_linear)
        
        bit_errors = 0
        frame_errors = 0
        total_bits = 0
        
        for frame in range(num_frames):
            # Generate random bits (all-zero codeword for simplicity, because LDPC is linear and we assume all codewords are equally likely)
            info_bits = np.random.randint(0, 2, n)
            # We assume we are using the all-zero codeword for simulation (but we need to encode properly)
            # For now, we assume the code is linear and we are sending the all-zero codeword.
            # In practice, we need to encode the info_bits to a codeword. Since we don't have the generator matrix, we skip encoding and assume all-zero.
            # This is valid for BER simulation if the code is linear and symmetric channel.
            codeword = np.zeros(n, dtype=int)
            # BPSK modulation: 0 -> +1, 1 -> -1
            transmitted_signal = 1 - 2 * codeword
            # AWGN channel
            noise = np.sqrt(noise_var) * np.random.randn(n)
            received_signal = transmitted_signal + noise
            # LLR from channel: for BPSK, LLR = 2 * received_signal / noise_var
            received_llr = 2 * received_signal / noise_var
            
            # Decode
            decoded_bits, success = bp_decode(H, received_llr, max_iter)
            
            # Count errors
            frame_errors += not success
            bit_errors += np.sum(decoded_bits != codeword)
            total_bits += n
        
        ber = bit_errors / total_bits
        fer = frame_errors / num_frames
        ber_list.append(ber)
        fer_list.append(fer)
        print(f"SNR: {snr_db} dB, BER: {ber}, FER: {fer}")
    
    return ber_list, fer_list

if __name__ == "__main__":
    n = 100
    dv = 3
    dc = 6
    snr_db_list = [0, 1, 2, 3, 4, 5]
    ber, fer = simulate_ldpc(n, dv, dc, snr_db_list, num_frames=100)
    
    plt.figure()
    plt.semilogy(snr_db_list, ber, 'o-', label='BER')
    plt.semilogy(snr_db_list, fer, 's-', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.title('LDPC Code Performance (n=100, dv=3, dc=6)')
    plt.savefig('../results/ber_fer_curve.png')
    plt.show()
