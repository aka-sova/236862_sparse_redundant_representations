import numpy as np

def get_pseudoinv(mx : np.array):
    # return pseudoinverse

    mx_t = np.transpose(mx)
    mx_t_mx = np.matmul(mx_t, mx)
    mx_t_mx_m1 = np.linalg.inv(mx_t_mx)

    return np.matmul(mx_t_mx_m1, mx_t)


#       def omp(A, b, k):
#       # OMP Solve the P0 problem via OMP
#       #
#       # Solves the following problem:
#       #   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
#       #
#       # The solution is returned in the vector x
#   
#           # Initialize the vector x
#           x = np.zeros(np.shape(A)[1])
#   
#           # TODO: Implement the OMP algorithm
#           
#   
#           # return the obtained x
#           return x


def omp(A : np.array, b: np.array, k : int = np.infty):
    """performing the OMP algorithm :
    input:
        A = atoms matrix
        b = synthetic generated signal
        k = number of atoms to look for in the solution
    output:
        x = sparse representation of the signal    
    """

    # print("\nSOLVING OMP\n-----------")

    x = [] # list of solutions
    r = [] # list of residuals
    S = [] # list of supports

    x.append(np.zeros((A.shape[1],1))) # initial solution\

    b = np.expand_dims(b, axis=1)
    r_0 = b - np.matmul(A,x[0])
    r.append(r_0)



    for k in range(0,min(k, A.shape[1])):
        
        
        # choose the atom
        A_t_rk_m1 = np.abs(np.matmul(np.transpose(A), r[-1]))

        atom_index = np.argmax(A_t_rk_m1)
        S.append(atom_index) # update support

        # solve LS problem to get the exact x values

        # 1. construct the A matrix from the selected columns
        A_k = A[:, S]
        A_pseudoinv = get_pseudoinv(A_k)
        x_k_S = np.matmul(A_pseudoinv, b)

        # create a new X vector with values only at the support
        x_k = np.zeros_like(x[0])
        x_k[S] = x_k_S

        x.append(x_k)

        r_k = b - np.matmul(A,x[-1])
        r.append(r_k)

        #   print(f"iteration #{k}\n------")
        #   print(f"For iteration {k}\nthe column chosen:{atom_index+1}")
        #   print(f"residual: {r_k}")
        #   print(f"residual norm: {np.linalg.norm(r_k)}")

    return x[-1].squeeze()