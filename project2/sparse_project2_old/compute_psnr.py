import numpy as np
import math

def compute_psnr(y_original, y_estimated):
    
    # COMPUTE_PSNR Computes the PSNR between two images
    #
    # Input:
    #  y_original  - The original image
    #  y_estimated - The estimated image
    #
    # Output:
    #  psnr_val - The Peak Signal to Noise Ratio (PSNR) score

    y_original = np.reshape(y_original,(-1))
    y_estimated = np.reshape(y_estimated,(-1))

    # Compute the dynamic range
    dynamic_range =    y_original.max() - y_original.min()


    # Compute the Mean Squared Error (MSE)
    mse_val = (np.linalg.norm(y_original-y_estimated)**2)/(40**2)


    # Compute the PSNR
    psnr_val = 10 * (2*np.log10(dynamic_range + 1e-6) - np.log10(mse_val + 1e-6))


    return psnr_val


