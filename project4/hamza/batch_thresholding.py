import numpy as np
import numpy.matlib

def batch_thresholding(D, Y, epsilon):
    # BATCH_THRESHOLDING Solve the pursuit problem via the error-constraint 
    # Thresholding pursuit
    # 
    # Solves the following problem:
    #   min_{alpha_i} \sum_i || alpha_i ||_0
    #                  s.t.  ||y_i - D alpha_i||_2**2 \leq epsilon**2 for all i, 
    # where D is a dictionary of size n X n, y_i are the input signals of
    # length n (being the columns of the matrix Y) and epsilon stands 
    # for the allowed residual error.
    #
    # The solution is returned in the matrix A, containing the representations 
    # of the patches as its columns, along with the denoised signals
    # given by  X = DA.
 
    
    # Get the number of atoms
    n = D.shape[1]
    
 
    # Get the number of patches
    N = Y.shape[1]


 
    # Compute the inner products between the dictionary atoms and the
    # input patches (hint: the result should be a matrix of size n X N)
    inner_products = D.T @ Y
    

 
    # Compute epsilon**2, which is the square residual error allowed per patch
    epsilon_sq = epsilon**2
 
    # Compute the square value of each entry in 'inner_products' matrix
    residual_sq = inner_products**2
 
    # Sort each column in 'residual_sq' matrix in ascending order
    mat_sorted = np.sort(residual_sq, axis=0)
    mat_inds = np.argsort(residual_sq, axis=0)


    # Compute the cumulative sums for each column of 'mat_sorted'
    # and save the result in the matrix 'accumulate_residual'
    accumulate_residual = np.cumsum(mat_sorted,axis=0)
    


 
    # Compute the indices of the dominant coefficients that we want to keep
    inds_to_keep = (accumulate_residual > epsilon_sq)
 
    # Allocate a matrix of size n X N to save the sparse vectors
    A = np.zeros([n,N])
    


 
    # In what follows we compute the location of each non-zero to be assigned 
    # to the matrix of the sparse vectors A. To this end, we need to map 
    # 'mat_inds' to a linear subscript format. The mapping will be done using 
    # Matlab's 'sub2ind' function.
 
    # Create a repetition of the column index for all rows
    col_sub = np.matlib.repmat(np.arange(N), n, 1)
 
    # Map the entries in 'inds_to_keep' to their corresponding locations
    # in 'mat_inds' and 'col_sub'.
    mat_inds_to_keep = mat_inds[inds_to_keep]
    col_sub_to_keep = col_sub[inds_to_keep]
    
    # Assign to 'A' the coefficients in 'inner_products' using
    # the precomputed 'mat_inds_to_keep' and 'col_sub_to_keep'
    A[mat_inds_to_keep,col_sub_to_keep] = inner_products[mat_inds_to_keep,col_sub_to_keep]
   
    
    
    # Reconstruct the patches using 'A' matrix
    X = D @ A 



    
    return X,A
 


