import numpy as np

def bp_decode(H, received_llr, max_iter=50):
    """
    Belief Propagation decoding for LDPC codes.
    Args:
        H: parity-check matrix (m x n)
        received_llr: initial LLRs from the channel (length n)
        max_iter: maximum number of iterations
    Returns:
        decoded_bits: hard decision after decoding (length n)
        success: whether the syndrome check is satisfied
    """
    m, n = H.shape
    # Initialize variable nodes
    var_nodes = received_llr.copy()
    # Message arrays: we'll use dictionaries for simplicity, but for performance we might use arrays.
    # Alternatively, we can use a matrix for messages from var to check and check to var.
    # Let's create two matrices: V2C and C2V of the same shape as H, but only for non-zero entries.
    # We'll use a list of lists for neighbors for each variable and check node.
    var_neighbors = [list(np.where(H[:, j] == 1)[0]) for j in range(n)]
    check_neighbors = [list(np.where(H[i, :] == 1)[0]) for i in range(m)]
    
    # Initialize V2C and C2V messages to zero for all edges.
    V2C = [ [0] * len(var_neighbors[j]) for j in range(n) ]
    C2V = [ [0] * len(check_neighbors[i]) for i in range(m) ]
    
    # Mapping for variable node j to the index in the check node i's neighbor list and vice versa.
    # We precompute the indices for each edge.
    # For each variable node j and its neighbor check node c (which is in var_neighbors[j]), we need to know the index of j in check_neighbors[c]
    # Let's create a structure for each variable node j: for each check neighbor, store the index in that check's neighbor list.
    var_to_check_index = [ [] for _ in range(n) ]
    for j in range(n):
        for c in var_neighbors[j]:
            var_to_check_index[j].append(check_neighbors[c].index(j))
    
    # Similarly, for each check node i and its variable neighbor j, we need the index of i in var_neighbors[j]
    check_to_var_index = [ [] for _ in range(m) ]
    for i in range(m):
        for v in check_neighbors[i]:
            check_to_var_index[i].append(var_neighbors[v].index(i))
    
    # Iterate
    for it in range(max_iter):
        # Step 1: Variable to Check messages
        for j in range(n):
            for idx, c in enumerate(var_neighbors[j]):
                # V2C[j][idx] = received_llr[j] + sum of all C2V messages to j except from c
                total = received_llr[j]
                for idx2, c2 in enumerate(var_neighbors[j]):
                    if c2 != c:
                        total += C2V[c2][check_to_var_index[c2][j]]  # Note: we need to map j to the index in c2's neighbor list
                V2C[j][idx] = total
        
        # Step 2: Check to Variable messages
        for i in range(m):
            for idx, v in enumerate(check_neighbors[i]):
                # Compute the check node message using the tanh rule (in log domain)
                # We use the approximation: tanh(x/2) = product of tanh( V2C[v][idx2] / 2 ) for all v' in N(i)\v
                # But in log-domain, we use the min-sum approximation for simplicity.
                signs = 1.0
                min_magnitude = float('inf')
                for idx2, v2 in enumerate(check_neighbors[i]):
                    if v2 != v:
                        message = V2C[v2][var_to_check_index[v2][i]]  # message from v2 to i
                        signs *= np.sign(message)
                        if abs(message) < min_magnitude:
                            min_magnitude = abs(message)
                # Min-sum approximation: magnitude is the minimum of the magnitudes of the incoming messages (excluding the one we are sending to)
                C2V[i][idx] = signs * min_magnitude
        
        # Step 3: Compute total LLR for each variable node and make hard decision
        total_LLR = received_llr.copy()
        for j in range(n):
            for idx, c in enumerate(var_neighbors[j]):
                total_LLR[j] += C2V[c][check_to_var_index[c][j]]
        
        decoded_bits = (total_LLR < 0).astype(int)
        
        # Check syndrome
        syndrome = np.dot(H, decoded_bits) % 2
        if np.sum(syndrome) == 0:
            return decoded_bits, True
    
    return decoded_bits, False
