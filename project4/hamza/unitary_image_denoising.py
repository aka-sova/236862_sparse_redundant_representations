import numpy as np    
from im2col import im2col
from col2im import col2im
from batch_thresholding import batch_thresholding
from compute_stat import compute_stat
from unitary_dictionary_learning import unitary_dictionary_learning
import matplotlib.pyplot as plt

def unitary_image_denoising(noisy_im, D_init, num_learning_iterations,epsilon):
    # UNITARY_IMAGE_DENOISING Denoise an image using unitary dictionary learning
    #
    # Inputs:
    #   noisy_im - The input noisy image
    #   D_init   - An initial UNITARY dictionary (e.g. DCT)
    #   epsilon  - The noise-level in a PATCH,
    #              used as the stopping criterion of the pursuit
    #
    # Outputs:
    #   est_unitary - The denoised image
    #   D_unitary   - The learned dictionary
    #   mean_error  - A vector, containing the average representation error,
    #                 computed per iteration and averaged over the total
    #                 training examples
    #   mean_cardinality - A vector, containing the average number of nonzeros,
    #                      computed per iteration and averaged over the total
    #                      training examples
    #


    #%% Dictionary Learning

    # Get the patch size [height, width] from D_init
    patch_size = (10,10)



    # Divide the noisy image into fully overlapping patches
    patches = im2col(noisy_im, patch_size, stepsize=1)


    # Train a dictionary via Procrustes analysis
    D_unitary, mean_error, mean_cardinality = unitary_dictionary_learning(patches, D_init, num_learning_iterations, epsilon)




    #%% Denoise the input image
    
    # TODO: Step 1: Compute the representation of each noisy patch using the
    # Thresholding pursuit
    est_patches, est_coeffs = batch_thresholding(D_unitary, patches, epsilon)

    # out of interest
    atoms_sums = np.sum(abs(est_coeffs), axis=1)
    atoms_sums = atoms_sums.reshape([10, -1])
    atoms_sums = atoms_sums.T

    plt.figure(4)
    plt.imshow(np.log10(atoms_sums), cmap='hot', interpolation='nearest')
    plt.savefig("imgs/dct_learned_heatmap.png")


    # TODO: Step 2: Reconstruct the image using 'col_to_im' function
    est_unitary = col2im(est_patches, patch_size, noisy_im.shape)




    #%% Compute and display the statistics

    print('\n\nUnitary dictionary: ')
    compute_stat(est_patches, patches, est_coeffs)

    return est_unitary, D_unitary, mean_error, mean_cardinality



