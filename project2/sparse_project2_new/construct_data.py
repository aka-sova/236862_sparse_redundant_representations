import numpy as np
import matplotlib.pyplot as plt

def construct_data(A, p, sigma, k):
    
    # CONSTRUCT_DATA Generate Mondrian-like synthetic image
    #
    # Input:
    #  A     - Dictionary of size (n**2 x m)
    #  p     - Percentage of known data in the range (0 1]
    #  sigma - Noise std
    #  k     - Cardinality of the representation of the synthetic image in the
    #          range [1 max(m,n)]
    #
    # Output:
    #  x0 - A sparse vector creating the Mondrian-like image b0
    #  b0 - The original image of size n^2
    #  noise_std  - The standard deviation of the noise added to b0
    #  b0_noisy   - A noisy version of b0 of size n^2
    #  C  - Sampling matrix of size (p*(n**2) x n^2), 0 < p <= 1
    #  b  - The corrupted image (noisy and subsampled version of b0) of size p*(n**2)
 
 
    # Get the size of the image and number of atoms
    n_squared, m = np.shape(A)
 
    # generate a Mondrian image
    # by drawing at random a sparse vector x0 of length m with cardinality k
 
    # Draw at random the locations of the non-zeros
    nnz_locs = np.random.choice(range(m), k, replace = False)

 
    # Draw at random the values of the coefficients
    nnz_vals = np.abs(np.random.randn(k))

 
    # Create a k-sparse vector x0 of length m given the nnz_locs and nnz_vals
    x0 = np.zeros(m)
    x0[nnz_locs] = nnz_vals

 
    # Given A and x0, compute the signal b0
    b0 = A[:,nnz_locs]@x0[nnz_locs]

 
    # Create the measured data vector b of size n^2
 
    # Compute the dynamic range
    dynamic_range = b0.max() - b0.min()

 
    # Create a noise vector
    noise_std = sigma * dynamic_range
    noise = noise_std * np.random.randn(n_squared)    
 
    # Add noise to the original image
    b0_noisy = b0 + noise

 
    # Create the sampling matrix C of size (p*n^2 x n^2), 0 < p <= 1
 
    # Create an identity matrix of size (n^2 x n^2)
    I = np.eye(n_squared)

 
    # Draw at random the indices of rows to be kept
    keep_inds = np.random.choice(range(n_squared), int(p*n_squared), replace = False)

 
    # Create the sampling matrix C of size (p*n^2 x n^2) by keeping rows
    # from I that correspond to keep_inds
    C = I[keep_inds,:]

 
    # Create a subsampled version of the noisy image
    b = b0_noisy[keep_inds]

    
    return x0, b0, noise_std, b0_noisy, C, b 


def compare_data(b_list = [], titles=[], fig_name = "default.png"):

    # reshape
    n = int(np.sqrt(b_list[0].shape[0]))

    fig = plt.figure()  # width, height in inches

    for i in range(len(b_list)):


        b_reshaped = np.reshape(b_list[i], [n, -1])
        b_reshaped *= 256
        b_reshaped = b_reshaped.astype(int)


        sub = fig.add_subplot(1, len(b_list), i+1)
        sub.imshow(b_reshaped, cmap='gray', interpolation='nearest')
        sub.set_title(titles[i])
        sub.xaxis.set_ticks([])
        sub.yaxis.set_ticks([])        

    plt.savefig(fig_name, dpi=300)



def display_data(b_clean, b_noisy, b_corrupted, C):

    # corrupted image

    # b : p          X (n**2)
    # C : p * (n**2) X (n**2)

    b_corrupted_full = np.transpose(C) @ b_corrupted

    # reshape
    n = int(np.sqrt(b_clean.shape[0]))

    b_clean_reshaped = np.reshape(b_clean, [n, -1])
    b_clean_reshaped *= 256
    b_clean_reshaped = b_clean_reshaped.astype(int)

    b_noisy_reshaped = np.reshape(b_noisy, [n, -1])
    b_noisy_reshaped *= 256
    b_noisy_reshaped = b_noisy_reshaped.astype(int)    

    b_corrupted_reshaped = np.reshape(b_corrupted_full, [n, -1])
    b_corrupted_reshaped *= 256
    b_corrupted_reshaped = b_corrupted_reshaped.astype(int)    


    fig = plt.figure(figsize=(3, 5))  # width, height in inches

    sub = fig.add_subplot(3, 1, 1)
    sub.imshow(b_clean_reshaped, cmap='gray', interpolation='nearest')
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])    

    sub = fig.add_subplot(3, 1, 2)
    sub.imshow(b_noisy_reshaped, cmap='gray', interpolation='nearest')
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])    

    sub = fig.add_subplot(3, 1, 3)
    sub.imshow(b_corrupted_reshaped, cmap='gray', interpolation='nearest')   
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])     

    plt.show()

    pass    


def show_comparison(b_clean, b_noisy, b_corrupted, C, b_recovered, fig_name = "default.png"):

    # corrupted image

    # b : p          X (n**2)
    # C : p * (n**2) X (n**2)

    b_corrupted_full = np.transpose(C) @ b_corrupted

    # reshape
    n = int(np.sqrt(b_clean.shape[0]))

    b_clean_reshaped = np.reshape(b_clean, [n, -1])
    b_clean_reshaped *= 256
    b_clean_reshaped = b_clean_reshaped.astype(int)

    b_noisy_reshaped = np.reshape(b_noisy, [n, -1])
    b_noisy_reshaped *= 256
    b_noisy_reshaped = b_noisy_reshaped.astype(int)    

    b_corrupted_reshaped = np.reshape(b_corrupted_full, [n, -1])
    b_corrupted_reshaped *= 256
    b_corrupted_reshaped = b_corrupted_reshaped.astype(int)    

    b_recovered_reshaped = np.reshape(b_recovered, [n, -1])
    b_recovered_reshaped *= 256
    b_recovered_reshaped = b_recovered_reshaped.astype(int)    
    


    fig = plt.figure()  # width, height in inches

    sub = fig.add_subplot(2, 2, 1)
    sub.imshow(b_clean_reshaped, cmap='gray', interpolation='nearest')
    sub.set_title("Clean")
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])

    sub = fig.add_subplot(2, 2, 2)
    sub.imshow(b_noisy_reshaped, cmap='gray', interpolation='nearest')
    sub.set_title("Noisy")
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])    

    sub = fig.add_subplot(2, 2, 3)
    sub.imshow(b_corrupted_reshaped, cmap='gray', interpolation='nearest')   
    sub.set_title("Corrupted")
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])    

    sub = fig.add_subplot(2, 2, 4)
    sub.imshow(b_recovered_reshaped, cmap='gray', interpolation='nearest')   
    sub.set_title("Recoveved")
    sub.xaxis.set_ticks([])
    sub.yaxis.set_ticks([])    


    plt.savefig(fig_name, dpi=300)

    pass    