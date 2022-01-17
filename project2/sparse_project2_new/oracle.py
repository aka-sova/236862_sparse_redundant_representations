import numpy as np

def oracle(CA, b, s):
    # ORACLE Implementation of the Oracle estimator
    #
    # Solves the following problem:
    #   min_x ||b - CAx||_2^2 s.t. supp{x} = s
    # where s is a vector containing the support of the true sparse vector
    #
    # The solution is returned in the vector x

    # Initialize the vector x
    x = np.zeros(np.shape(CA)[1])

    # Implement the Oracle estimator
    x[s] = np.linalg.pinv(CA[:,s].squeeze())@b


    return x