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

    # TODO: Compute the dynamic range
    # Write your code here... dynamic_range = ????


    # TODO: Compute the Mean Squared Error (MSE)
    # Write your code here... mse_val = ????


    # TODO: Compute the PSNR
    # Write your code here... psnr_val = ????


    return psnr_val

