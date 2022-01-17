import numpy as np
from PIL import Image
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
    n = int(np.sqrt(n_squared))
 
    # generate a Mondrian image
    # by drawing at random a sparse vector x0 of length m with cardinality k
 
    # Draw at random the locations of the non-zeros
    nnz_locs = np.random.choice(range(m), k, replace = False)

 
    # Draw at random the values of the coefficients
    # nnz_vals = np.random.uniform(k)
    
    # FIX : draw at random the value of each entry from a Gaussian distribution with zero-mean and standard-deviation that equals to 1
    nnz_vals = np.abs(np.random.normal(loc = 0, scale=1, size=k))

 
    # Create a k-sparse vector x0 of length m given the nnz_locs and nnz_vals
    x0 = np.zeros(m)
    x0[nnz_locs] = nnz_vals

 
    # Given A and x0, compute the signal b0
    b0 = A[:,nnz_locs] @ x0[nnz_locs]

    # Create the measured data vector b of size n^2
 
    # Compute the dynamic range
    dynamic_range = b0.max() - b0.min()

 
    # Create a noise vector
    noise_std = sigma * dynamic_range
    # noise = noise_std * np.random.randn(n**2)    
    noise = np.random.normal(loc = 0, scale=noise_std, size=n**2)
 
    # Add noise to the original image
    b0_noisy = b0 + noise

    # debug - show histograms of the original image and the noise
    if False:
        fig = plt.figure(figsize=(15, 5))  # width, height in inches

        sub1 = fig.add_subplot(2, 1, 1)
        sub1.hist(b0)

        sub2 = fig.add_subplot(2, 1, 2, sharex=sub1)
        sub2.hist(noise)
        plt.show()    
 

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


def compare_data(b_clean, b_reconstructed):

    # reshape
    n = int(np.sqrt(b_clean.shape[0]))

    b_clean_reshaped = np.reshape(b_clean, [n, -1])
    b_clean_reshaped *= 256
    b_clean_reshaped = b_clean_reshaped.astype(int)

    b_reconstructed_reshaped = np.reshape(b_reconstructed, [n, -1])
    b_reconstructed_reshaped *= 256
    b_reconstructed_reshaped = b_reconstructed_reshaped.astype(int)


    fig = plt.figure(figsize=(3, 5))  # width, height in inches

    sub = fig.add_subplot(2, 1, 1)
    sub.imshow(b_clean_reshaped, cmap='gray', interpolation='nearest')

    sub = fig.add_subplot(2, 1, 2)
    sub.imshow(b_reconstructed_reshaped, cmap='gray', interpolation='nearest')

    plt.show()
    pass


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

    sub = fig.add_subplot(3, 1, 2)
    sub.imshow(b_noisy_reshaped, cmap='gray', interpolation='nearest')

    sub = fig.add_subplot(3, 1, 3)
    sub.imshow(b_corrupted_reshaped, cmap='gray', interpolation='nearest')    

    plt.show()

    pass

def compare_weights(x0, x_oracle, s):

    fig = plt.figure(figsize=(5, 5))  # width, height in inches

    plt.plot(x0[s], 'bo')
    plt.plot(x_oracle[s], 'ro')
    plt.grid()
    plt.legend(["true weights", "oracle weights"])
    plt.show() 
    pass     
