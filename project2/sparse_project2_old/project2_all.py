import numpy as np
import matplotlib.pyplot as plt
from construct_data import construct_data, display_data, compare_data, compare_weights
from compute_psnr import compute_psnr
from compute_effective_dictionary import compute_effective_dictionary
from omp import omp
from bp_admm import bp_admm
from oracle import oracle

# In this project we will solve a variant of the P0^\epsilon for filling-in 
# missing pixels (also known as "inpainting") in a synthetic image.
 
#%% Parameters
 
# Set the size of the desired image is (n x n)
n = 40


# Set the number of atoms
m = 2*n**2


# Set the percentage of known data
p = 1


# Set the noise std
sigma = 0.05 # FIXED


# Set the cardinality of the representation
true_k = 10


# Base seed - A non-negative integer used to reproduce the results
# Set an arbitrary value for base_seed
base_seed = 1234


# Run the different algorithms for num_experiments and average the results
num_experiments = 10
 
 
#%% Create a dictionary A of size (n**2 x m) for Mondrian-like images
 
# initialize A with zeros
A = np.zeros([n**2, m])

 
# In this part we construct A by creating its atoms one by one, where
# each atom is a rectangle of random size (in the range 5-20 pixels),
# position (uniformly spread in the area of the image), and sign. 
# Lastly we will normalize each atom to a unit norm.
for i in range(A.shape[1]):

    # Choose a specific random seed to reproduce the results
    np.random.seed(i + base_seed)
    empty_atom_flag = 1
    
    while empty_atom_flag:
        
        # Create a rectangle of random size and position
        size_x, size_y = np.random.randint(5,21,2)
        x_start = np.random.randint(0, n-size_x)
        y_start = np.random.randint(0, n-size_y)
        atom = np.zeros([n, n])
        atom[x_start:x_start+size_x,y_start:y_start+size_y] = np.random.uniform(0,1)*np.ones([size_x, size_y])


        # Reshape the atom to a 1D vector
        atom = np.reshape(atom,(-1))
        
        # Verify that the atom is not empty or nearly so
        if np.sqrt(np.sum(atom**2)) > 1e-5:
            empty_atom_flag = 0
            
            # Normalize the atom
            atom = atom/np.linalg.norm(atom)
            
            
            
            # Assign the generated atom to the matrix A
            A[:,i] = atom
            
 
#%% Oracle Inpainting

# Allocate a vector to store the PSNR results
PSNR_oracle = np.zeros(num_experiments)
 
# Loop over num_experiments
for experiment in range(num_experiments):
    
    # Choose a specific random seed to reproduce the results
    np.random.seed(experiment + 1 + base_seed)
    
    # Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

    # debug
    # display_data(b0, b0_noisy, b, C)
    
    # Compute the subsampled dictionary
    s = x0.nonzero()
    A_eff_normalized, atoms_norm = compute_effective_dictionary(C, A)
    
    
    # Compute the oracle estimation
    x_oracle = oracle(A_eff_normalized, b, s)
    
    # Un-normalize the coefficients
    x_oracle = x_oracle / atoms_norm
    
    compare_weights(x0, x_oracle, s)

    # Compute the estimated image    
    b_oracle = A @ x_oracle

    # compare_data(b0, b_oracle)

    # Compute the PSNR
    PSNR_oracle[experiment] = compute_psnr(b0, b_oracle)
    
    # Print some statistics
    print('Oracle experiment %d/%d, PSNR: %.3f' % (experiment+1,num_experiments,PSNR_oracle[experiment]))

# Display the average PSNR of the oracle
print('Oracle: Average PSNR = %.3f\n' % np.mean(PSNR_oracle))
 
#%% Greedy: OMP Inpainting
""" 
# We will sweep over k = 1 up-to k = max_k and pick the best result
max_k = min(2*true_k, m)
 
# Allocate a vector to store the PSNR estimations per each k
PSNR_omp = np.zeros((num_experiments,max_k))
 
# Loop over the different realizations
for experiment in range(num_experiments):
    
    # Choose a specific random seed to reproduce the results
    np.random.seed(experiment+1 + base_seed)
    
    # Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)
    
    # Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)
   
    # Run the OMP for various values of k and pick the best results
    for k_ind in range(max_k):
        
        # Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, k_ind+1)
        
        # Un-normalize the coefficients
        x_omp = x_omp/atoms_norm
        
        # Compute the estimated image        
        b_omp = A @ x_omp
        
        # Compute the current PSNR
        PSNR_omp[experiment, k_ind] = compute_psnr(b0, b_omp)
        
        # Save the best result of this realization, we will present it later
        if PSNR_omp[experiment, k_ind] == max(PSNR_omp[experiment, :]) :
            best_b_omp = b_omp
            
        # Print some statistics
        print('OMP experiment %d/%d, cardinality %d, PSNR: %.3f' % (experiment+1,num_experiments,k_ind,PSNR_omp[experiment,k_ind]))
 
# Compute the best PSNR, computed for different values of k
PSNR_omp_best_k = np.max(PSNR_omp,axis=-1)
 
# Display the average PSNR of the OMP (obtained by the best k per image)
print('OMP: Average PSNR = %.3f\n' % np.mean(PSNR_omp_best_k))
 
# Plot the average PSNR vs. k
psnr_omp_k = np.mean(PSNR_omp, 0)
k_scope = np.arange(1, max_k + 1)
plt.figure(1)
plt.plot(k_scope, psnr_omp_k, '-r*')
plt.xlabel("k", fontsize=16)
plt.ylabel("PSNR [dB]", fontsize=16)
plt.title("OMP: PSNR vs. k, True Cardinality = " + str(true_k)) 
 
"""
###

#%% Convex relaxation: Basis Pursuit Inpainting via ADMM
 
# We will sweep over various values of lambda
num_lambda_values = 10
 
# Allocate a vector to store the PSNR results obtained for the best lambda
PSNR_admm_best_lambda = np.zeros(num_experiments)
 
# Loop over num_experiments
for experiment in range(num_experiments):
    
    # Choose a specific random seed to reproduce the results
    np.random.seed(experiment+1 + base_seed)
    
    # Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)
    
    # Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)
    
    # Run the BP for various values of lambda and pick the best result
    lambda_max = np.linalg.norm(A_eff_normalized.T @ b, np.inf)
    lambda_vec = np.logspace(-5,0,num_lambda_values)*lambda_max    
    psnr_admm_lambda = np.zeros(num_lambda_values)
    
    # Loop over various values of lambda
    for lambda_ind in range(num_lambda_values):
        
        # Compute the BP estimation
        x_admm = bp_admm(A_eff_normalized, b, lambda_vec[lambda_ind])
        
        # Un-normalize the coefficients
        x_admm = x_admm / atoms_norm
        
        # Compute the estimated image        
        b_admm = A @ x_admm
        
        # Compute the current PSNR
        psnr_admm_lambda[lambda_ind] = compute_psnr(b0, b_admm)
        
        # Save the best result of this realization, we will present it later
        if psnr_admm_lambda[lambda_ind] == max(psnr_admm_lambda) :
            best_b_admm = b_admm
        
        # print some statistics
        print('BP experiment %d/%d, lambda %d/%d, PSNR %.3f' % \
              (experiment+1,num_experiments,lambda_ind+1,num_lambda_values,psnr_admm_lambda[lambda_ind]))
    
    # Save the best PSNR
    PSNR_admm_best_lambda[experiment] = max(psnr_admm_lambda)
    
 
# Display the average PSNR of the BP
print("BP via ADMM: Average PSNR = ", np.mean(PSNR_admm_best_lambda), "\n")
 
# Plot the PSNR vs. lambda of the last realization
plt.figure(2)
plt.semilogx(lambda_vec, psnr_admm_lambda, '-*r')
plt.xlabel(r'$\lambda$', fontsize=16)
plt.ylabel("PSNR [dB]", fontsize=16)
plt.title("BP via ADMM: PSNR vs. " + r'$\lambda$')
 
#%% show the results
 
# Show the images obtained in the last realization, along with their PSNR
plt.figure(3,figsize=(10,6))

plt.subplot(231)
plt.imshow(np.reshape(b0,(n, n)), cmap='gray')
plt.title("Original Image, k = " + str(true_k))

plt.subplot(232)
plt.imshow(np.reshape(b0_noisy,(n, n)), cmap='gray')
plt.title("Noisy Image, PSNR = " + str(compute_psnr(b0, b0_noisy)))

plt.subplot(233)
corrupted_img = np.reshape(C.T @ b,(n, n))
plt.imshow(corrupted_img, cmap='gray')
plt.title("Corrupted Image, PSNR = " + str(compute_psnr(b0, C.T @ b)))

plt.subplot(234)
plt.imshow(np.reshape(b_oracle,(n, n)), cmap='gray')
plt.title("Oracle, PSNR = " + str(compute_psnr(b0, b_oracle)))
"""
plt.subplot(235)
plt.imshow(np.reshape(best_b_omp,(n, n)), cmap='gray')
plt.title("OMP, PSNR = " + str(compute_psnr(b0, best_b_omp)))
"""
plt.subplot(236)
plt.imshow(np.reshape(best_b_admm,(n, n)), cmap='gray')
plt.title("BP-ADMM, PSNR = " + str(compute_psnr(b0, best_b_admm)))
 
#%% Compare the results

# Show a bar plot of the average PSNR value obtained per each algorithm
plt.figure(4)
mean_psnr = np.array([np.mean(PSNR_oracle), np.mean(PSNR_admm_best_lambda)]) #Replace with: mean_psnr = np.array([np.mean(PSNR_oracle), np.mean(PSNR_omp_best_k), np.mean(PSNR_admm_best_lambda)])
x_bar = np.arange(2) #Replace with: x_bar = np.arange(3)
plt.bar(x_bar, mean_psnr)
plt.xticks(x_bar, ('Oracle', 'BP-ADMM')) #Replace with: plt.xticks(x_bar, ('Oracle', 'BP-ADMM'))
plt.ylabel('PSNR [dB]', fontsize=16)
plt.xlabel('Algorithm', fontsize=16)
"""
#%% Run OMP with fixed cardinality and increased percentage of known data
 
# TODO: Set the noise std
# Write your code here... sigma = ????



# TODO: Set the cardinality of the representation
# Write your code here... true_k = ????



# TODO: Create a vector of increasing values of p in the range [0.4 1]. The
# length of this vector equal to num_values_of_p = 7.
# Write your code here... num_values_of_p = ???? p_vec = ????




# We will repeat the experiment for num_experiments realizations
num_experiments = 100
 
# Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_p = np.zeros(num_values_of_p)
 
# Loop over num_experiments
for experiment in range(num_experiments):
    
    # Loop over various values of p
    for p_ind in range(num_values_of_p):
        
        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment+1 + base_seed)
        
        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p_vec[p_ind], sigma, true_k)
                
        # Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)
        
        # Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k)
        
        # Un-normalize the coefficients
        x_omp = x_omp/atoms_norm
        
        # Compute the estimated image        
        b_omp = A @ x_omp
                
        # TODO: Compute the MSE of the estimate
        # Write your code here... cur_mse = ????

        
                
        # Compute the current normalized MSE and aggregate
        mse_omp_p[p_ind] = mse_omp_p[p_ind] + cur_mse / (dynamic_range*noise_std)**2
        
        #print some statistics
        print('OMP as a function p: experiment %d/%d, p_ind %d/%d, MSE: %.3f' % \
              (experiment+1,num_experiments,p_ind+1,num_values_of_p,mse_omp_p[p_ind]))
 
 
# Compute the average PSNR over the different realizations
mse_omp_p = mse_omp_p / num_experiments
 
# Plot the average normalized MSE vs. p
plt.figure(5)
plt.plot(p_vec, mse_omp_p, '-*r')
plt.ylabel('Normalized-MSE', fontsize=16)
plt.xlabel('p', fontsize=16)
plt.title('OMP with k = ' + str(true_k) + ', Normalized-MSE vs. p') 
 
 
#%% Run OMP with fixed cardinality and increased noise level
 
# TODO: Set the cardinality of the representation
# Write your code here... true_k = ????



# TODO: Set the percentage of known data
# Write your code here... p = ????



# TODO: Create a vector of increasing values of sigma in the range [0.15 0.5].
# The length of this vector equal to num_values_of_sigma = 10.
# Write your code here... num_values_of_sigma = ???? sigma_vec = ????



# Repeat the experiment for num_experiments realizations
num_experiments = 100
 
# Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_sigma = np.zeros(num_values_of_sigma)
 
# Loop over num_experiments
for experiment in range(num_experiments):
    
    # Loop over increasing noise level
    for sigma_ind in range(num_values_of_sigma):
        
        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment+1 + base_seed)
        
        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma_vec[sigma_ind], true_k)
        
        # Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)
        
        # Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k)
        
        # Un-normalize the coefficients
        x_omp = x_omp/atoms_norm
        
        # Compute the estimated image        
        b_omp = A @ x_omp
                
        # TODO: Compute the MSE of the estimate
        # Write your code here... cur_mse = ????

        
        
        # Compute the current normalized MSE and aggregate
        mse_omp_sigma[sigma_ind] = mse_omp_sigma[sigma_ind] + cur_mse / (noise_std**2)
        
        # Print some statistics
        print('OMP as a function sigma: experiment %d/%d, sigma_ind %d/%d, MSE: %.3f' % \
              (experiment+1,num_experiments,sigma_ind+1,num_values_of_sigma,mse_omp_sigma[sigma_ind]))
 
 
# Compute the average PSNR over the different realizations
mse_omp_sigma = mse_omp_sigma / num_experiments
    
# Plot the average normalized MSE vs. sigma
plt.figure(6)
plt.plot(sigma_vec, mse_omp_sigma, '-*r')
plt.ylim(0.5*min(mse_omp_sigma), 5*max(mse_omp_sigma))
plt.ylabel('Normalized-MSE')
plt.xlabel('sigma')
plt.title("OMP with k = " + str(true_k) + ", Normalized-MSE vs. sigma")

plt.show()


"""
print()