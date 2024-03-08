import torch
import hw4_utils
import numpy as np
def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [N, 2].
        init_c: initial centroids, shape [2, 2]. Each row is a centroid.

    Return:
        c: shape [2, 2]. Each row is a centroid.
    """
    

    if X is None:
        X, init_c = hw4_utils.load_data()
    samples, features = X.shape
    c1, c2 = map(np.array, init_c)
    
    
    
    #initialize
    count = 0
    p_c1, p_c2 = torch.zeros(2), torch.zeros(2)
    
    
    while True:
        # Assign the points using vectorized computation
        distances_to_c1 = torch.norm(X - c1, dim=1)
        distances_to_c2 = torch.norm(X - c2, dim=1)
        c1_list = X[distances_to_c1 < distances_to_c2].tolist()
        c2_list = X[distances_to_c1 >= distances_to_c2].tolist()

        # Update previous centeroids
        p_c1, p_c2 = c1.copy(), c2.copy()

        # Visualize clusters for the first iteration
        if count == 0:
            hw4_utils.vis_cluster(torch.tensor(np.array([c1, c2])), torch.tensor(c1_list), torch.tensor(c2_list))

        # Update centers
        c1 = np.mean(c1_list, axis=0) if c1_list else c1
        c2 = np.mean(c2_list, axis=0) if c2_list else c2
        
        count += 1

        # Visualize clusters after re-centering
        hw4_utils.vis_cluster(torch.tensor(np.array([c1, c2])), torch.tensor(c1_list), torch.tensor(c2_list))

        # Check for convergence
        if np.allclose(p_c1, c1) and np.allclose(p_c2, c2):
            break

    print(count - 1)    
    Objective = 0
    for i in range(samples):
        point = X[i].numpy()
        if point.tolist() in c1_list:
            Objective += np.linalg.norm(point - c1) * 0.5
        else:
            Objective += np.linalg.norm(point - c2) * 0.5
    print(Objective)
    # Return the final centroids as a tensor
    result = torch.tensor(np.array([c1, c2]))
    return result

#call function
k_means()

    