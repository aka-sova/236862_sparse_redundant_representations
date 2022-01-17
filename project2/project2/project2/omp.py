import numpy as np

def omp(CA, b, k):
    
    # OMP Solve the sparse coding problem via OMP
    #
    # Solves the following problem:
    #   min_x ||b - CAx||_2^2 s.t. ||x||_0 <= k
    #
    # The solution is returned in the vector x

    # Initialize the vector x
    x = np.zeros(np.shape(CA)[1])

    # CA : [p*n**2 X m] = [640 X 3200]
    # b  : [p*n**2,]    = [640,]
    # x  : [m, ]        = [3200, ]


    S = [] # list of supports

    # x.append(np.zeros((A.shape[1],1))) # initial solution\
    r = b - np.matmul(CA, x)



    for _ in range(k):
        
        
        # choose the atom
        A_t_rk_m1 = np.abs(np.matmul(np.transpose(CA), r))

        atom_index = np.argmax(A_t_rk_m1)
        S.append(atom_index) # update support

        # solve LS problem to get the exact x values

        # 1. construct the A matrix from the selected columns
        A_k = CA[:, S]
        A_pseudoinv = np.linalg.pinv(A_k)
        x_k_S = np.matmul(A_pseudoinv, b)

        # create a new X vector with values only at the support
        x_k = np.zeros_like(x)
        x_k[S] = x_k_S

        # replace current x with new calculated
        x = x_k

        # recalculate the residual
        r = b - np.matmul(CA,x)

        #   print(f"iteration #{k}\n------")
        #   print(f"For iteration {k}\nthe column chosen:{atom_index+1}")
        #   print(f"residual: {r_k}")
        #   print(f"residual norm: {np.linalg.norm(r_k)}")

    return x 