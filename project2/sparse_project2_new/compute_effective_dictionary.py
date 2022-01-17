import numpy as np

def compute_effective_dictionary(C, A):
    
    # COMPUTE_EFFECTIVE_DICTIONARY Computes the subsampled and normalized
    #   dictionary
    #
    # Input:
    #  C     - Sampling matrix of size (p*(n**2) x n^2)
    #  A     - Dictionary of size ((n**2) x m)
    #
    # Output:
    #  A_eff_normalized - The subsampled and normalized dictionary of size (p*(n**2) x m)
    #  atoms_norm - A vector of length m, containing the norm of each sampled atom
    

    A_eff = C@A
    # print(len(support))

    # Compute the norm of each atom
    atoms_norm = np.linalg.norm(A_eff, axis=0)
    
    # Normalize the columns of A_eff, avoid division by zero
    A_eff_normalized = A_eff
    non_zero_norms = (atoms_norm > 0)
    A_eff_normalized[:,non_zero_norms] = A_eff[:,non_zero_norms]/atoms_norm[non_zero_norms]

    
    return A_eff_normalized, atoms_norm