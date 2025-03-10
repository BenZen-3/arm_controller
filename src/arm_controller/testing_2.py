import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state

# GMM STUFF

def sample_from_distribution(prob_dist, n_samples=10000, random_state=None):
    """
    Sample points from a 2D probability distribution.
    
    Parameters:
    ----------
    prob_dist : numpy.ndarray
        2D array representing probability distribution
    n_samples : int, optional
        Number of points to sample (default: 10000)
    random_state : int or None, optional
        Random seed for reproducibility
    
    Returns:
    -------
    tuple
        (x_samples, y_samples) coordinates
    """
    # Get grid dimensions
    height, width = prob_dist.shape
    
    # Ensure probabilities are positive and sum to 1
    prob_flat = prob_dist.flatten()
    prob_flat = np.maximum(prob_flat, 0)
    prob_flat = prob_flat / np.sum(prob_flat)
    
    # Generate random indices based on probability distribution
    rng = check_random_state(random_state)
    indices = rng.choice(len(prob_flat), size=n_samples, p=prob_flat)
    
    # Convert flat indices to 2D coordinates
    y_indices = indices // width
    x_indices = indices % width
    
    # Map indices to actual coordinate values on the grid
    x_min, x_max = -5, 5  # Assuming your default plot range
    y_min, y_max = -5, 5
    
    x_samples = x_min + (x_max - x_min) * x_indices / (width - 1)
    y_samples = y_min + (y_max - y_min) * y_indices / (height - 1)
    
    return x_samples, y_samples

def fit_gmm_to_distribution(prob_dist, n_components=10, n_samples=10000, random_state=None):
    """
    Fit a Gaussian Mixture Model to a probability distribution.
    
    Parameters:
    ----------
    prob_dist : numpy.ndarray
        2D array representing probability distribution
    n_components : int, optional
        Number of Gaussian components (default: 10)
    n_samples : int, optional
        Number of points to sample (default: 10000)
    random_state : int or None, optional
        Random seed for reproducibility
    
    Returns:
    -------
    tuple
        (fitted GMM, list of gaussian parameters)
    """
    # Sample points from the distribution
    x_samples, y_samples = sample_from_distribution(prob_dist, n_samples, random_state)
    samples = np.column_stack([x_samples, y_samples])
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state
    )
    gmm.fit(samples)
    
    # Extract parameters in the format used by your existing code
    gaussian_params = []
    for i in range(n_components):
        mean_x, mean_y = gmm.means_[i]
        covariance = gmm.covariances_[i]
        
        # Extract variance and correlation
        sigma_x = np.sqrt(covariance[0, 0])
        sigma_y = np.sqrt(covariance[1, 1])
        rho = covariance[0, 1] / (sigma_x * sigma_y)
        
        # Add weight from GMM
        weight = gmm.weights_[i]
        
        gaussian_params.append((mean_x, mean_y, sigma_x, sigma_y, rho, weight))
    
    return gmm, gaussian_params

def evaluate_gmm_on_grid(gmm, mesh_grid):
    """
    Evaluate a fitted GMM on a mesh grid.
    
    Parameters:
    ----------
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM
    mesh_grid : tuple
        Tuple containing (X, Y) meshgrid arrays
    
    Returns:
    -------
    numpy.ndarray
        GMM probability density evaluated on the grid
    """
    X, Y = mesh_grid
    xy_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Get log probabilities from GMM
    log_probs = gmm.score_samples(xy_points)
    
    # Convert to probabilities and reshape to grid
    probs = np.exp(log_probs).reshape(X.shape)
    
    # Normalize to [0, 1] range
    if np.max(probs) > 0:
        probs = probs / np.max(probs)
    
    return probs

# OLDER STUFF BELOW HERE

def calculate_gaussian_field(mean_x, mean_y, cov_matrix, mesh_grid):
    """
    Calculate a 2D Gaussian distribution on a given meshgrid.
    
    Parameters:
    ----------
    mean_x : float
        Mean (center) x-coordinate
    mean_y : float
        Mean (center) y-coordinate
    cov_matrix : numpy.ndarray
        2x2 covariance matrix
    mesh_grid : tuple
        Tuple containing (X, Y) meshgrid arrays
    
    Returns:
    -------
    numpy.ndarray
        Array of Gaussian probability values
    """
    X, Y = mesh_grid
    pos = np.dstack((X, Y))
    
    # Create the multivariate normal distribution
    mean = np.array([mean_x, mean_y])
    rv = multivariate_normal(mean, cov_matrix)
    
    # Evaluate the PDF on the grid
    gaussian = rv.pdf(pos)
    
    return gaussian

def calculate_rectangle_sdf(center_x, center_y, width, height, mesh_grid, 
                           truncate=True, smoothing_sigma=0):
    """
    Calculate the signed distance field (SDF) for a rectangle.
    
    Parameters:
    ----------
    center_x : float
        X-coordinate of the rectangle's center
    center_y : float
        Y-coordinate of the rectangle's center
    width : float
        Width of the rectangle
    height : float
        Height of the rectangle
    mesh_grid : tuple
        Tuple containing (X, Y) meshgrid arrays
    truncate : bool, optional
        Whether to truncate SDF values to [-1, 1] range (default: True)
    smoothing_sigma : float, optional
        Sigma parameter for Gaussian smoothing (default: 0, no smoothing)
    
    Returns:
    -------
    numpy.ndarray
        SDF array
    """
    X, Y = mesh_grid
    
    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2
    
    # Calculate the distance to the rectangle for each point
    # For points outside the rectangle
    dx = np.maximum(np.abs(X - center_x) - half_width, 0)
    dy = np.maximum(np.abs(Y - center_y) - half_height, 0)
    outside_distance = np.sqrt(dx**2 + dy**2)
    
    # Calculate the inside distance (negative values)
    inside_x = half_width - np.abs(X - center_x)
    inside_y = half_height - np.abs(Y - center_y)
    inside_distance = -np.minimum(inside_x, inside_y)
    inside_distance[inside_x < 0] = 0
    inside_distance[inside_y < 0] = 0
    
    # Combine inside and outside distances to get the SDF
    sdf = np.where((np.abs(X - center_x) <= half_width) & 
                  (np.abs(Y - center_y) <= half_height),
                  inside_distance, outside_distance)
    
    # Apply smoothing if requested
    if smoothing_sigma > 0:
        sdf = gaussian_filter(sdf, sigma=smoothing_sigma)
    
    # Truncate values if requested
    if truncate:
        sdf = np.clip(sdf, -1, 1)
    
    return sdf

def plot_field(mesh_grid, field, rectangles=None, gaussians=None, 
              cmap=cm.viridis, show_3d=True, title=None):
    """
    Plot a field (SDF, Gaussian, or combined).
    
    Parameters:
    ----------
    mesh_grid : tuple
        Tuple containing (X, Y) meshgrid arrays
    field : numpy.ndarray
        The field array to plot
    rectangles : list of tuples, optional
        List of (center_x, center_y, width, height) tuples for rectangles to draw
    gaussians : list of tuples, optional
        List of (mean_x, mean_y) tuples for Gaussian centers to mark
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the plot (default: viridis)
    show_3d : bool, optional
        Whether to show the 3D surface plot (default: True)
    title : str, optional
        Custom title for the plot
    
    Returns:
    -------
    matplotlib.figure.Figure
        The figure object
    """
    X, Y = mesh_grid
    
    # Create figure
    if show_3d:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot the field as a contour plot
    contour = ax1.contourf(X, Y, field, 50, cmap=cmap)
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Field Visualization')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    fig.colorbar(contour, ax=ax1, label='Field Value')
    
    # Draw contour lines at specific levels
    contour_levels = np.linspace(np.min(field), np.max(field), 9)
    contour_lines = ax1.contour(X, Y, field, 
                              levels=contour_levels,
                              colors='white', linestyles='dashed')
    ax1.clabel(contour_lines, inline=True, fontsize=8)
    
    # Draw rectangles if provided
    if rectangles is not None:
        for rect in rectangles:
            center_x, center_y, width, height = rect
            half_width = width / 2
            half_height = height / 2
            rect_x = center_x - half_width
            rect_y = center_y - half_height
            rectangle = plt.Rectangle((rect_x, rect_y), width, height, 
                                    edgecolor='white', facecolor='none', linewidth=2)
            ax1.add_patch(rectangle)
    
    # Mark Gaussian centers if provided
    if gaussians is not None:
        for mean in gaussians:
            ax1.plot(mean[0], mean[1], 'ro', markersize=5)
    
    # Plot the field as a 3D surface if requested
    if show_3d:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax2.plot_surface(X, Y, field, cmap=cmap, antialiased=True)
        if title:
            ax2.set_title(f'3D View of {title}')
        else:
            ax2.set_title('3D View of Field')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Field Value')
    
    plt.tight_layout()
    return fig

def create_mesh_grid(grid_size=100, plot_range=(-5, 5)):
    """
    Create a meshgrid for evaluation of fields.
    
    Parameters:
    ----------
    grid_size : int, optional
        Resolution of the grid (default: 100)
    plot_range : tuple, optional
        Range of x and y coordinates to compute (default: (-5, 5))
    
    Returns:
    -------
    tuple
        (X grid, Y grid)
    """
    x = np.linspace(plot_range[0], plot_range[1], grid_size)
    y = np.linspace(plot_range[0], plot_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def create_covariance_matrix(sigma_x, sigma_y, rho):
    """
    Create a 2x2 covariance matrix from standard deviations and correlation.
    
    Parameters:
    ----------
    sigma_x : float
        Standard deviation in x direction
    sigma_y : float
        Standard deviation in y direction
    rho : float
        Correlation coefficient between x and y (-1 to 1)
    
    Returns:
    -------
    numpy.ndarray
        2x2 covariance matrix
    """
    return np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])

def normalize_field(field, target_range=(-1, 1)):
    """
    Normalize a field to a target range.
    
    Parameters:
    ----------
    field : numpy.ndarray
        Field to normalize
    target_range : tuple, optional
        Target range for normalization (default: (-1, 1))
    
    Returns:
    -------
    numpy.ndarray
        Normalized field
    """
    min_val, max_val = target_range
    field_min, field_max = np.min(field), np.max(field)
    
    # Handle the case where field is constant
    if field_max == field_min:
        return np.full_like(field, (min_val + max_val) / 2)
    
    # Scale to [0, 1]
    normalized = (field - field_min) / (field_max - field_min)
    
    # Scale to target range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def combine_distributions(distributions, method='sum', normalize=True):
    """
    Combine multiple probability-like distributions into a single distribution.
    
    Parameters:
    ----------
    distributions : list
        List of numpy arrays representing probability-like distributions
    method : str, optional
        Method for combining distributions:
        - 'sum': Add distributions (union-like behavior)
        - 'product': Multiply distributions (intersection-like behavior)
        - 'max': Take element-wise maximum (union)
        - 'min': Take element-wise minimum (intersection)
    normalize : bool, optional
        Whether to normalize the result to [0, 1] range (default: True)
    
    Returns:
    -------
    numpy.ndarray
        Combined distribution
    """
    if not distributions:
        raise ValueError("No distributions provided")
    
    # Check that all distributions have the same shape
    shape = distributions[0].shape
    for dist in distributions:
        if dist.shape != shape:
            raise ValueError("All distributions must have the same shape")
    
    # Combine distributions according to the specified method
    if method == 'sum':
        combined = np.sum(distributions, axis=0)
    elif method == 'product':
        combined = np.ones(shape)
        for dist in distributions:
            combined *= dist
    elif method == 'max':
        combined = np.maximum.reduce(distributions)
    elif method == 'min':
        combined = np.minimum.reduce(distributions)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to [0, 1] if requested
    if normalize and np.max(combined) > 0:
        combined = combined / np.max(combined)
    
    return combined

def prepare_sdf_for_combination(sdf_list):
    """
    Convert a list of SDFs to probability-like distributions ready for combination.
    
    Parameters:
    ----------
    sdf_list : list
        List of SDF arrays
    
    Returns:
    -------
    list
        List of probability-like distributions
    """
    prob_distributions = []
    
    for sdf in sdf_list:
        # Convert SDF to probability-like (higher inside, lower outside)
        prob = -sdf  # Invert so inside is positive
        prob = (prob + 1) / 2  # Scale to [0, 1]
        prob_distributions.append(prob)
    
    return prob_distributions

def create_multiple_rectangle_distribution(rectangle_params, mesh_grid, combination_method='sum'):
    """
    Create a combined probability distribution from multiple rectangles.
    
    Parameters:
    ----------
    rectangle_params : list of tuples
        List of (center_x, center_y, width, height, smoothing) tuples for rectangles
    mesh_grid : tuple
        Tuple containing (X, Y) meshgrid arrays
    combination_method : str, optional
        Method for combining distributions (default: 'sum')
    
    Returns:
    -------
    tuple
        (Combined probability distribution, List of rectangle parameters without smoothing)
    """
    # Calculate SDF for each rectangle
    sdf_list = []
    rectangle_info = []
    
    for rect in rectangle_params:
        # Unpack parameters (handle both 4 and 5 parameter versions)
        if len(rect) == 5:
            center_x, center_y, width, height, smoothing = rect
        else:
            center_x, center_y, width, height = rect
            smoothing = 0.0
        
        # Calculate SDF
        sdf = calculate_rectangle_sdf(
            center_x, center_y, width, height, mesh_grid, 
            truncate=True, smoothing_sigma=smoothing
        )
        
        sdf_list.append(sdf)
        rectangle_info.append((center_x, center_y, width, height))
    
    # Convert SDFs to probability distributions
    prob_distributions = prepare_sdf_for_combination(sdf_list)
    
    # Combine distributions
    combined_prob = combine_distributions(prob_distributions, method=combination_method)
    
    return combined_prob, rectangle_info

# Example usage:
def main_with_multiple_rectangles():
    """
    Main function to create and fit GMM to multiple rectangle SDFs.
    """
    # Set up the grid
    grid_size = 400
    plot_range = (-5, 5)
    mesh_grid = create_mesh_grid(grid_size, plot_range)
    
    # Define multiple rectangles: (center_x, center_y, width, height, smoothing)
    rectangle_params = [
        (0, 0, 1.0, 2.0, 0.2),
        (1, 1, 3.0, 0.5, 0.2),
        (-2, -1, 1.5, 1.0, 0.1),
        # Add more rectangles as needed
    ]
    
    # Create combined probability distribution
    combined_prob, rectangle_info = create_multiple_rectangle_distribution(
        rectangle_params, mesh_grid, combination_method='sum'
    )
    
    # Fit GMM to the distribution
    n_components = 120  # Adjust based on complexity
    gmm, gaussian_params = fit_gmm_to_distribution(combined_prob, n_components=n_components, n_samples=n_components*1000)
    
    # Evaluate the fitted GMM on the grid
    gmm_prob = evaluate_gmm_on_grid(gmm, mesh_grid)
    
    # Plot the original combined distribution
    fig1 = plot_field(
        mesh_grid, combined_prob,
        rectangles=rectangle_info,
        title="Combined Rectangle Distribution"
    )
    
    # Plot the GMM approximation
    gaussian_centers = [(g[0], g[1]) for g in gaussian_params]
    fig2 = plot_field(
        mesh_grid, gmm_prob,
        gaussians=gaussian_centers,
        title="GMM Approximation with {} components".format(n_components)
    )
    
    # Plot the difference (error)
    error = np.abs(combined_prob - gmm_prob)
    fig3 = plot_field(
        mesh_grid, error,
        title="Approximation Error"
    )
    
    # Print the learned Gaussian parameters
    print("\nLearned Gaussian Parameters:")
    print("-----------------------------")
    for i, (mean_x, mean_y, sigma_x, sigma_y, rho, weight) in enumerate(gaussian_params):
        if weight > 0.01:  # Only show significant components
            print(f"Gaussian {i+1}:")
            print(f"  Mean: ({mean_x:.2f}, {mean_y:.2f})")
            print(f"  Sigma: ({sigma_x:.2f}, {sigma_y:.2f})")
            print(f"  Correlation: {rho:.2f}")
            print(f"  Weight: {weight:.4f}")
    
    # Calculate overall error metrics
    mse = np.mean(error**2)
    max_error = np.max(error)
    print(f"\nError Metrics:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Maximum Error: {max_error:.6f}")
    
    plt.show()

if __name__ == "__main__":
    main_with_multiple_rectangles()