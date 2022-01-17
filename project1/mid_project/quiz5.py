
import numpy as np
from numpy.lib.shape_base import apply_over_axes
from numpy.linalg import solve

A = np.array(([0.1817, 0.5394, -0.1197, 0.6404], 
              [0.6198, 0.1994, 0.0946, -0.3121],
              [-0.7634, -0.8181, 0.9883, 0.7018]))

b = np.array(([1.1862],[-0.1158], [-0.1093]))


def get_pseudoinv(mx : np.array):
    # return pseudoinverse

    mx_t = np.transpose(mx)
    mx_t_mx = np.matmul(mx_t, mx)
    mx_t_mx_m1 = np.linalg.inv(mx_t_mx)

    return np.matmul(mx_t_mx_m1, mx_t)


def solve_OMP(A : np.array, b: np.array, num_steps : int = np.infty):
    # performing the OMP algorithm

    print("\nSOLVING OMP\n-----------")

    x = [] # list of solutions
    r = [] # list of residuals
    S = [] # list of supports

    x.append(np.zeros((4,1))) # initial solution\

    r_0 = b - np.matmul(A,x[0])
    r.append(r_0)



    for k in range(0,min(num_steps, A.shape[1])):
        
        
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

        print(f"iteration #{k}\n------")
        print(f"For iteration {k}\nthe column chosen:{atom_index+1}")
        print(f"residual: {r_k}")
        print(f"residual norm: {np.linalg.norm(r_k)}")

    return x[-1]


def solve_LS_OMP(A : np.array, b: np.array, num_steps : int = np.infty):
    # performing the OMP algorithm

    print("\nSOLVING LS OMP\n-----------")

    x = [] # list of solutions
    r = [] # list of residuals
    S = [] # list of supports

    x.append(np.zeros((4,1))) # initial solution

    r_0 = b - np.matmul(A,x[0])
    r.append(r_0)



    for k in range(0,min(num_steps, A.shape[1])):
        
        
        # choose the atom by solving a list of LS problems

        # 1. create list of supports
        supports_l = list(range(0, A.shape[1])) # indexes

        err_l = []
        err = np.infty
        best_support = None
        best_x_k = None

        # sweep through all of the supports
        for supp in supports_l:
            
            # 2. Augment the supports list with new support
            #       if the item is already in the list, don't add
            if supp not in S:
                curr_supp = S + [supp]
            else:
                curr_supp = S

            # build the augmented A_s matrix with updated support
            curr_A = A[:, curr_supp]

            # solve the LS problem
            A_pseudoinv = get_pseudoinv(curr_A)
            x_k_S = np.matmul(A_pseudoinv, b)

            # create a new X vector with values only at the current support
            x_k_current = np.zeros_like(x[0])
            x_k_current[curr_supp] = x_k_S

            # compute the error
            curr_err = np.linalg.norm(b - np.matmul(A, x_k_current))

            if curr_err < err:

                best_x_k = x_k_current
                best_support = supp
                err = curr_err

        atom_index = best_support

        S.append(atom_index) # update support

        # solve LS problem to verify
        # note : no need to solve the LS problem, it was already solved

        # 1. construct the A matrix from the UPDATED support
        # A_k = A[:, S]
        # A_pseudoinv = get_pseudoinv(A_k)
        # x_k_S = np.matmul(A_pseudoinv, b)

        # create a new X vector with values only at the support
        # x_k = np.zeros_like(x[0])
        # x_k[S] = x_k_S

        x.append(best_x_k)

        # calculate residual for record
        r_k = b - np.matmul(A,x[-1])
        r.append(r_k)

        print(f"\niteration #{k}\n------")
        print(f"For iteration {k}\nthe column chosen:{atom_index+1}")
        print(f"residual: {r_k}")
        print(f"residual norm: {np.linalg.norm(r_k)}")

    return x[-1]


def solve_WMP(A : np.array, b: np.array, t : int = 0.5, num_steps : int = np.infty):
    # performing the OMP algorithm

    print("\nSOLVING WMP\n-----------")

    x = [] # list of solutions
    r = [] # list of residuals
    S = [] # list of supports

    x.append(np.zeros((4,1))) # initial solution\

    r_0 = b - np.matmul(A,x[0])
    r.append(r_0)

    # maximum value of the A column * residual
    A_T = np.transpose(A)

    for k in range(0,min(num_steps, A.shape[1])):
        
        # threshold abs val
        curr_max_val = np.max(np.abs(r[-1])) * t
        
        # choose the first column that crosses the threshold
        for i in range(0, A.shape[1]):

            energy_lvl = np.abs(np.matmul(A_T[i], r[-1]))

            if energy_lvl > curr_max_val:
                atom_index = i
                break



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

        print(f"iteration #{k}\n------")
        print(f"For iteration {k}\nthe column chosen:{atom_index+1}")
        print(f"residual: {r_k}")
        print(f"residual norm: {np.linalg.norm(r_k)}")

    return x[-1]


def solve_thresholding(A : np.array, b: np.array, num_steps : int = np.infty):
    # performing the OMP algorithm

    print("\nSOLVING Thresholding\n-----------")

    x = [] # list of solutions
    r = [] # list of residuals
    S = [] # list of supports

    x.append(np.zeros((4,1))) # initial solution\

    r_0 = b - np.matmul(A,x[0])
    r.append(r_0)

    # compute the order of adding new atoms
    A_t_b = np.abs(np.matmul(np.transpose(A), b))
    indexes_decreasing_list = list(np.argsort(A_t_b.shape[0] - 1 - np.transpose(A_t_b))[0])



    for k in range(0,min(num_steps, A.shape[1])):
        
        
        # choose the atom
        atom_index = indexes_decreasing_list[k]

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

        print(f"iteration #{k}\n------")
        print(f"For iteration {k}\nthe column chosen:{atom_index+1}")
        print(f"residual: {r_k}")
        print(f"residual norm: {np.linalg.norm(r_k)}")

    return x[-1]



x_omp = solve_OMP(A,b, num_steps=2)   
x_ls_omp = solve_LS_OMP(A,b, num_steps=2)
x_wmp = solve_WMP(A,b, t=0.5, num_steps=2)
x_thr = solve_thresholding(A,b, num_steps=2)







