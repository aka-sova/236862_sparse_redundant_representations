import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from compute_psnr import compute_psnr
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from show_dictionary import show_dictionary
from dct_image_denoising import dct_image_denoising
from unitary_image_denoising import unitary_image_denoising

import os
os.makedirs('imgs',exist_ok=True)

#%% Part A: Data construction
 
# Read an image
orig_im = np.array(Image.open('barbara.png'))
# Crop the image
orig_im = orig_im[:256, 251:251+256]

# Set the seed for the random generator
seed = 42

 
# Set a fixed random seed to reproduce the results
np.random.seed(seed)
 
# Set the standard-deviation of the Gaussian noise
sigma = 20


 
# Add noise to the input image
noisy_im = orig_im + np.random.randn(256,256)*sigma



# Compute the PSNR of the noisy image and print its value
psnr_noisy = compute_psnr(orig_im, noisy_im)
print('PSNR of the noisy image is %.3f\n\n\n' % psnr_noisy)
 
# Show the original and noisy images
plt.figure(0, figsize=[10,10])
plt.subplot(2,2,1)
plt.imshow(orig_im,'gray')
plt.axis('off')
plt.title('Original')
plt.subplot(2,2,2)
plt.imshow(noisy_im,'gray')
plt.axis('off')
plt.title('Noisy PSNR = %.3f' % psnr_noisy)
 
#%% Part B: Patch-Based Image Denoising
 
# Set the patch dimensions [height, width]
patch_size = (10,10)
 
# TODO: Create a unitary DCT dictionary
D_DCT = build_dct_unitary_dictionary(patch_size)


 
# Show the unitary DCT dictionary
plt.figure(1,figsize=[8,4])
plt.subplot(1, 2, 1)
show_dictionary(D_DCT)
plt.title('Unitary DCT Dictionary') 

 
#%% Part B-1: Unitary DCT denoising
 
# Set the noise-level of a PATCH for the pursuit,
# multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
epsilon_dct = sigma * patch_size[0] * 1.1**0.5

 
# Denoise the input image via the DCT transform
est_dct = dct_image_denoising(noisy_im, D_DCT, epsilon_dct)
 
# Compute and print the PSNR
psnr_dct = compute_psnr(orig_im, est_dct)
print('DCT dictionary: PSNR %.3f\n\n\n' % psnr_dct)
 
# Show the resulting image
plt.figure(0)
plt.subplot(2,2,3)
plt.imshow(est_dct,'gray')
plt.axis('off')
plt.title('DCT: $\epsilon$ = %.3f PSNR = %.3f' % (epsilon_dct,psnr_dct))
 
#%% Part B-2: Unitary dictionary learning for image denoising
 
# Set the number of training iterations for the learning algorithm
T = 20


 
# Set the noise-level of a PATCH for the pursuit,
# multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
epsilon_learning = epsilon_dct


 
# Denoise the image using unitary dictionary learning
est_learning, D_learned, mean_error, mean_cardinality = unitary_image_denoising(\
    noisy_im, D_DCT, T, epsilon_learning)
 
#% Show the dictionary
plt.figure(1)
plt.subplot(1,2,2)
show_dictionary(D_learned) 
plt.title('Learned Unitary Dictionary')
plt.savefig("imgs/fig1_dicts.png", dpi=300)
 
#% Show the representation error and the cardinality as a function of the
#% learning iterations
plt.figure(2, figsize=[8,4])
plt.subplot(1,2,1) 
plt.plot(np.arange(T), mean_error, linewidth=2.0)
plt.ylim((0, 5*int(np.sqrt(patch_size[0]*patch_size[1]))*sigma))
plt.ylabel('Average Representation Error')
plt.xlabel('Learning Iteration')
plt.subplot(1,2,2)
plt.plot(np.arange(T),mean_cardinality, linewidth=2.0)
plt.ylabel('Average Number of Non-Zeros')
plt.xlabel('Learning Iteration')
plt.savefig("imgs/fig2_learning_dict.png", dpi=300)
 
#% Compute and print the PSNR
psnr_unitary = compute_psnr(orig_im, est_learning)
print('Unitary dictionary: PSNR %.3f\n\n\n' % psnr_unitary)



# # PSNR vs iterations
# psnr_lst = []
# for t in range(T):
#     t_learning, _, _, _ = unitary_image_denoising(noisy_im, D_DCT, t, epsilon_learning)

#     psnr_t = compute_psnr(orig_im, t_learning)
#     psnr_lst.append(psnr_t)

# plt.figure(5)
# plt.plot(np.arange(T), psnr_lst, linewidth=2.0)
# plt.grid(True)
# plt.ylabel('PSNR')
# plt.xlabel('Learning Iteration')
# plt.savefig("imgs/fig5_psnr_learning.png", dpi=300)

 
#% Show the results
plt.figure(0)
plt.subplot(2,2,4)
plt.imshow(est_learning,'gray')
plt.axis('off')
plt.title('Unitary: $\epsilon$ = %.3f PSNR = %.3f' % (epsilon_learning, psnr_unitary))
plt.savefig("imgs/fig0_barbaras.png", dpi=300)
 
#%% SOS Boosting
 
# Set the strengthening factor
rho = 1


 
# Set the noise-level in a PATCH for the pursuit. 
# A common choice is a slightly higher noise-level than the one set 
# in epsilon_learning, e.g. 1.1*epsilon_learning;
epsilon_sos = 1.1*epsilon_learning



# Init D_sos to be D_learned 
D_sos = D_learned 


 
# Compute the signal-strengthen image by adding to 'noisy_im' the 
# denoised image 'est_learning', multiplied by an appropriate 'rho'
s_im = noisy_im + rho*est_learning


 
# Operate the denoiser on the signal-strengthened image
est_learning_sos = unitary_image_denoising(s_im, D_sos, T, epsilon_sos)[0]



# Subtract from 'est_learning_sos' image the previous estimate
# 'est_learning', multiplied by the strengthening factor
est_learning_sos = est_learning_sos - rho*est_learning


 
# Compute and print the PSNR
psnr_unitary_sos = compute_psnr(orig_im, est_learning_sos)
print('SOS Boosting: epsilon %.3f, rho %.2f, PSNR %.3f\n\n\n' % \
      (epsilon_sos,rho,psnr_unitary_sos))


plt.figure(7)
plt.subplot(1,2,1)
plt.imshow(est_learning,'gray')
plt.axis('off')
plt.title('Unitary: $\epsilon$ = %.3f PSNR = %.3f' % (epsilon_learning, psnr_unitary))
plt.subplot(1,2,2)
plt.imshow(est_learning_sos,'gray')
plt.axis('off')
plt.title('Unitary SOS: $\epsilon$ = %.3f PSNR = %.3f' % (epsilon_sos, psnr_unitary_sos))
plt.savefig("imgs/fig7_sos_restored.png", dpi=300)


plt.figure(8)
plt.title("Differences between the Learned and SOS boosted restoration")
plt.imshow(256-abs(est_learning - est_learning_sos),'gray')
plt.axis('off')
plt.savefig("imgs/fig8_differences.png", dpi=300)


plt.show()