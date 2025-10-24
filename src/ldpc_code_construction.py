import numpy as np

def generate_regular_ldpc(n, dv, dc):
    """
    Generate a regular LDPC code parity-check matrix H of size (m x n) with variable degree dv and check degree dc.
    Note: m = n * dv / dc must be an integer.
    """
    m = int(n * dv / dc)
    H = np.zeros((m, n), dtype=int)
    
    # We use a simple random construction without avoiding 4-cycles for simplicity.
    # Fill the matrix such that each column has dv ones and each row has dc ones.
    col_count = np.zeros(n, dtype=int)
    row_count = np.zeros(m, dtype=int)
    
    # We'll create a list of edges and then place them in H
    edges = []
    for i in range(n):
        edges.extend([i] * dv)  # each variable node i appears dv times
    # Now we assign these edges to check nodes (each check node must have dc edges)
    # We shuffle the edges and then assign to check nodes in order
    np.random.shuffle(edges)
    check_assignments = []
    for j in range(m):
        check_assignments.extend([j] * dc)
    
    for j, i in zip(check_assignments, edges):
        H[j, i] = 1
    
    return H

def get_tanner_graph(H):
    """
    Construct the Tanner graph from the parity-check matrix H.
    Returns:
        variable_nodes: list of variable nodes (each is an index)
        check_nodes: list of check nodes (each is an index)
        edges: list of tuples (variable_index, check_index)
    """
    m, n = H.shape
    edges = []
    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                edges.append((j, i))  # variable j connected to check i
    return list(range(n)), list(range(m)), edges
