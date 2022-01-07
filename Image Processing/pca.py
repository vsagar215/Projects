from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# Loads the dataset from a provided .npy file, re-centers it around 
# the origin and returns it as a NumPy array of floats
def load_and_center_dataset(filename):
    x = np.load(filename)  # Adding image data as a n X d matrix
    
    mu_x = np.mean(x, axis=0)  # Computing mean
    x_cent = x - mu_x  # Subtracting means
    
    return x_cent

# Calculates and returns the covariance matrix of the dataset as 
# a NumPy matrix (d x d array)
def get_covariance(dataset): 
    factor = ( len(dataset) - 1) # Factor to divide by
    
    covar_mat = np.dot(np.transpose(dataset),dataset) / factor # x transpose dot x
    
    return np.array(covar_mat)  

# Performs eigen decomposition on the covariance matrix and returns a diagonal matrix (NumPy array) 
# with the largest m eigenvalues on the diagonal in descending order and a matrix (NumPy array)
# with the corresponding eigenvectors as columns
def get_eig(S, m):
    d = len(S) # size of the covar matrix
    eigen_values, eigen_vectors = eigh(
        S, subset_by_index=[d-m, d-1]) # returns the 'm' largest eigen vals
    
    eigen_values = np.diagflat(np.flip(eigen_values)) # flip it into desc order and measure along the diag
    eigen_vectors = np.flip(eigen_vectors, 1) # flip the eigen vectors along the cols
    
    return np.array(eigen_values), np.array(eigen_vectors) # return the NumPy arrays

# Returns all eigenvalues and corresponding eigenvectors in similar format as get_eig that 
# explain more than perc % of variance
def get_eig_perc(S, perc):
    variance = np.sum(eigh(S, eigvals_only = True))
    variance = variance*perc

    values, vectors = eigh(S, subset_by_value = [variance, np.inf])

    values = np.diagflat(np.flip(values))
    vectors = np.flip(vectors, 1)

    return np.array(values), np.array(vectors)

# project each (d x 1) image into your m-dimensional subspace (spanned by m vectors of size d x 1)
# and return the new representation as a d x 1 NumPy array
def project_image(image, U):
    alpha = np.dot(np.transpose(U), image)
    
    x_pro = np.dot(U, alpha)

    return np.array(x_pro)

# use matplotlib to display a visual representation of the original image 
# and the projected image side-by-side
def display_image(orig, proj):

    # Step 1: Reshaping images
    orig = np.reshape(orig, (32,32), order='F')
    proj = np.reshape(proj, (32,32), order='F')
    
    # Step 2: Creating fig with 1 row 2 sub plots
    fig, (a1,a2) = plt.subplots(1, 2)

    # Step 3: Setting titles
    a1.set_title('Original')    
    a2.set_title('Projection')

    # Step 4: Rendering orig and proj
    orig_pic = a1.imshow(orig, aspect = 'equal')
    proj_pic = a2.imshow(proj, aspect = 'equal')

    # Step 5: Creating colorbars
    fig.colorbar(orig_pic, ax=a1)
    fig.colorbar(proj_pic, ax=a2)
    
    plt.show()