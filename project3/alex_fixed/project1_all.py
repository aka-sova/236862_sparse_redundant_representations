import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import view_as_windows as viewW
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from show_dictionary import show_dictionary
from batch_thresholding import batch_thresholding
from compute_stat import compute_stat
from unitary_dictionary_learning import unitary_dictionary_learning

os.makedirs('imgs',exist_ok=True)

#%% Part A: Data Construction and Parameter-Setting

# Read an image
im = np.array(Image.open('barbara.png'))

# Show the image
plt.figure(0)
plt.imshow(im,'gray')
plt.title('Original image')
plt.savefig("imgs/1_barbara.png", dpi=300)

# Patch dimensions [height, width]
patch_size = [6,6]

# TODO: Divide the image into FULLY overlapping patches using skimage viewW function
# Write your code here... all_patches = ???
all_patches = viewW(im, patch_size, step=1) # (507, 507, 6, 6)

# all_patches[0,0] = im[0:6, 0:6]

# we need of size [n**2 X (512-n+1)**2]
all_patches = all_patches.reshape(507**2, 36).T


# Number of patches to train on
NUM_TRAINING_PATCHES = 10000
# Number of patches to test on
NUM_TEST_PATCHES = 5000

# TODO: Set the SEED for the random generator
# Write your code here... SEED = ???
SEED = 42


# Set a fixed random SEED to reproduce the results
np.random.seed(SEED)

# TODO: Create a training set by choosing a random subset
# of 'NUM_TRAINING_PATCHES' taken from 'all_patches'
# Write your code here... train_patches = ???

random_train_locs = np.random.choice(range(all_patches.shape[1]), NUM_TRAINING_PATCHES, replace = False)
train_patches = all_patches[:,random_train_locs]


# TODO: Create a test set by choosing another random subset
# of 'NUM_TEST_PATCHES' taken from the remaining patches
# Write your code here... test_patches = ???

# remove all the patches chosen for training using sets
all_indexes = np.linspace(0, all_patches.shape[1], all_patches.shape[1], dtype=int)
remaining_patches_indexes = np.array(list(set(all_indexes) - set(random_train_locs)))

# choose from the remaining indexes the locations in the original array
random_test_indexes = np.random.choice(range(remaining_patches_indexes.shape[0]), NUM_TEST_PATCHES, replace = False)
random_test_original_locs = remaining_patches_indexes[random_test_indexes]

test_patches = all_patches[:, random_test_original_locs]

# TODO: Initialize the dictionary
# Write your code here... D_DCT = build_dct_unitary_dictionary( ? )

D_DCT = build_dct_unitary_dictionary([6,6])

# Show the unitary DCT dictionary
plt.figure(1)
plt.subplot(1, 2, 1)
show_dictionary(D_DCT)
plt.title('Unitary DCT Dictionary')
plt.savefig("imgs/2_dct_dict.png", dpi=300)

# TODO: Set K - the cardinality of the solution.
# This will serve us later as the stopping criterion of the pursuit
# Write your code here... K = ???

K = 4



#%% Part B: Compute the Representation Error Obtained by the DCT Dictionary

# Compute the representation of each patch that belongs to the training set using Thresholding
est_train_patches_dct, est_train_coeffs_dct = batch_thresholding(D_DCT, train_patches, K)

# Compute the representation of each patch that belongs to the test set using Thresholding
est_test_patches_dct, est_test_coeffs_dct = batch_thresholding(D_DCT, test_patches, K)

# Compute and display the statistics
print('\n\nDCT dictionary: Training set, ')
compute_stat(est_train_patches_dct, train_patches, est_train_coeffs_dct)
print('DCT dictionary: Testing  set, ')
compute_stat(est_test_patches_dct, test_patches, est_test_coeffs_dct)
print('\n\n')

#%% Part C: Procrustes Dictionary Learning

# TODO: Set the number of training iterations
# Write your code here... T = ???

T = 200

# Train a dictionary via Procrustes analysis
D_learned, mean_error, mean_cardinality = unitary_dictionary_learning(train_patches, D_DCT, T, K)

# Show the dictionary
plt.figure(1)
plt.subplot(1, 2, 2)
show_dictionary(D_learned)
plt.title("Learned Unitary Dictionary")
plt.savefig("imgs/3_unitary_dict.png", dpi=300)

# Show the representation error and the cardinality as a function of the learning iterations
plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(np.arange(T), mean_error, linewidth=2.0)
plt.ylabel("Average Representation Error")
plt.xlabel("Learning Iteration")
plt.subplot(1, 2, 2)
plt.plot(np.arange(T), mean_cardinality, linewidth=2.0)
plt.ylabel('Average Number of Non-Zeros')
plt.ylim((K-1, K+1))
plt.xlabel('Learning Iteration')
plt.savefig("imgs/4_nonzeros.png", dpi=300)

# Compute the representation of each signal that belong to the training set using Thresholding
est_train_patches_learning, est_train_coeffs_learning = batch_thresholding(D_learned, train_patches, K)

# Compute the representation of each signal that belong to the testing set using Thresholding
est_test_patches_learning, est_test_coeffs_learning = batch_thresholding(D_learned, test_patches, K)

# Compute and display the statistics
print('\n\nLearned dictionary: Training set, ')
compute_stat(est_train_patches_learning, train_patches, est_train_coeffs_learning)
print('Learned dictionary: Testing  set, ')
compute_stat(est_test_patches_learning, test_patches, est_test_coeffs_learning)
print('\n\n') 
