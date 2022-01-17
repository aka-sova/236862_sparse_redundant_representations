# In this project we demonstrate the OMP and BP algorithms, by running them 
# on a set of signals and checking whether they provide the desired outcome
import numpy as np
from numpy.core.defchararray import replace
from omp import omp
from lp import lp
import matplotlib.pyplot as plt
import datetime
 
#%% Parameters
 
# Setting the length of the signal

n = 50 

# Setting the number of atoms in the dictionary

m = 100

# Setting the maximum number of non-zeros in the generated vector

s_max = 20


# Setting the minimal entry value

min_coeff_val = 1


# Setting the maximal entry value

max_coeff_val = 3


# Number of realizations

num_realizations = 200

# Base seed: A non-negative integer used to reproduce the results
# Setting an arbitrary value for base seed

base_seed = 10

 
#%% Create the dictionary
 
# Creating a random matrix A of size (n x m)
A = np.random.rand(n,m)

 
# Normalizing the columns of the matrix to have a unit norm
normalization_term = np.linalg.norm(A,axis=0)
A = A/normalization_term
 
#%% Create data and run OMP and BP
 
# Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4
# Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4

# Allocate a matrix to save the L2 error of the obtained solutions
L2_error = np.zeros((s_max,num_realizations,2))
# Allocate a matrix to save the support recovery score
support_error = np.zeros((s_max,num_realizations,2))

time_calc = np.zeros((s_max,num_realizations,2))
           
# Loop over the sparsity level
for s in range(s_max):

    s = s+1
    # Use the same random seed in order to reproduce the results if needed
    np.random.seed(s + base_seed)

    print(f"Ininitializing for s = {s}")
    
    # Loop over the number of realizations
    for experiment in range(num_realizations):

        # print(f"Experiment: {experiment} / {num_realizations}")
   
        # In this part we will generate a test signal b = A_normalized @ x by 
        # drawing at random a sparse vector x with s non-zeros entries in 
        # true_supp locations with values in the range of [min_coeff_val, max_coeff_val]
        
        x = np.zeros(m)
        
        # Drawing at random a true_supp vector
        # true_supp =  np.random.randint(0,m,s)
        true_supp =  np.random.choice(m,s, replace=False) # without repetitions
        
        
        # Drawing at random the coefficients of x in true_supp locations
        x[true_supp] = np.random.uniform(min_coeff_val,max_coeff_val,len(true_supp))
        x[true_supp] = x[true_supp]*((-1)**np.random.randint(0,2,len(true_supp)))       
        
        
        # Creating the signal b
        b = A@x
        
        omp_init = datetime.datetime.now()

        # Run OMP
        x_omp = omp(A, b, s)

        omp_elapsed = datetime.datetime.now() - omp_init
        time_calc[s-1, experiment, 0] = omp_elapsed.total_seconds()

                
        # Compute the relative L2 error
        L2_error[s-1,experiment,0] = np.linalg.norm(x-x_omp) / np.linalg.norm(x)
        
        # Get the indices of the estimated support
        estimated_supp = np.nonzero(abs(x_omp) > eps_coeff)
        
        # Compute the support recovery error
        support_error[s-1,experiment,0] = 1 - len(np.intersect1d(estimated_supp,true_supp))/max(len(estimated_supp[0]),len(estimated_supp[0]))
        
        lp_init = datetime.datetime.now()      

        # Run BP
        x_lp = lp(A, b, tol_lp)
        
        lp_elapsed = datetime.datetime.now() - lp_init
        time_calc[s-1, experiment, 1] = lp_elapsed.total_seconds()          
        
        # Compute the relative L2 error
        L2_error[s-1,experiment,1] = np.linalg.norm(x-x_lp) / np.linalg.norm(x)
        
        
        # Getting the indices of the estimated support, where the
        # coeffecients are larger (in absolute value) than eps_coeff
        # Write your code here... estimated_supp = ????
        estimated_supp = np.nonzero(abs(x_lp) > eps_coeff)
        
        # Compute the support recovery score
        support_error[s-1,experiment,1] = 1 - len(np.intersect1d(estimated_supp,true_supp))/max(len(estimated_supp[0]),len(estimated_supp[0]))

 
#%% Display the results 
plt.rcParams.update({'font.size': 14})
# Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,0],axis=1),color='red')  ### Remove '#' when finishing OMP code
plt.plot(np.arange(s_max)+1, np.mean(L2_error[:s_max,:,1],axis=1),color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.axis((0,s_max,0,1))
plt.legend(['OMP','LP'])
plt.grid()
plt.savefig("Average and Relative L2-Error.png", dpi=300)

# Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,0],axis=1),color='red') ### Same as above
plt.plot(np.arange(s_max)+1, np.mean(support_error[:s_max,:,1],axis=1),color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.axis((0,s_max,0,1))
plt.legend(['OMP','LP'])
plt.grid()
plt.savefig("Probability of Error in Support.png", dpi=300)


# Plot the average time required to run each algorithm
plt.figure()
plt.plot(np.arange(s_max)+1, np.mean(time_calc[:s_max,:,0],axis=1),color='red') ### Same as above
plt.plot(np.arange(s_max)+1, np.mean(time_calc[:s_max,:,1],axis=1),color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Time elapsed to run each single algorithmon average [seconds]')
plt.legend(['OMP','LP'])
plt.grid()
plt.savefig("Time elapsed.png", dpi=300)

plt.show()

print("hey")