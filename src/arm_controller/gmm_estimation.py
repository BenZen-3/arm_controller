import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation
import time


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability-distribution-like objects.
    
    All probability distributions should be able to:
    1. Return their probability density values over a meshgrid
    2. Sample points from their distribution
    3. Be combined with other distributions
    """
    
    def __init__(self, mesh_grid=None, grid_size=100, plot_range=(-5, 5)):
        """
        Initialize a probability distribution.
        
        Parameters:
        ----------
        mesh_grid : tuple, optional
            Tuple containing (X, Y) meshgrid arrays. If None, will create a new meshgrid.
        grid_size : int, optional
            Resolution of the grid if creating a new meshgrid (default: 100)
        plot_range : tuple, optional
            Range of x and y coordinates if creating a new meshgrid (default: (-5, 5))
        """
        if mesh_grid is None:
            self.mesh_grid = self.create_mesh_grid(grid_size, plot_range)
        else:
            self.mesh_grid = mesh_grid
        
        # The cached probability density distribution
        self._probability = None
    
    @staticmethod
    def create_mesh_grid(grid_size=100, plot_range=(-5, 5)):
        """
        Create a meshgrid for evaluation of fields.
        
        Parameters:
        ----------
        grid_size : int
            Resolution of the grid
        plot_range : tuple
            Range of x and y coordinates to compute
        
        Returns:
        -------
        tuple
            (X grid, Y grid)
        """

        if plot_range[0] >= plot_range[1]:
            raise ValueError(f"The mesh grid range is wrong. Range: {plot_range}")

        x = np.linspace(plot_range[0], plot_range[1], grid_size)
        y = np.linspace(plot_range[0], plot_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        return X, Y
    
    @abstractmethod
    def get_probability(self):
        """
        Return the probability density distribution.
        
        Returns:
        -------
        numpy.ndarray
            2D array of probability values
        """
        pass
    
    def sample_points(self, n_samples=10000):
        """
        Sample points from the probability distribution.
        
        Parameters:
        ----------
        n_samples : int, optional
            Number of points to sample (default: 10000)

        Returns:
        -------
        tuple
            (x_samples, y_samples) coordinates
        """
        prob_dist = self.get_probability()
        X, Y = self.mesh_grid
        
        # Get grid dimensions
        height, width = prob_dist.shape
        
        # Ensure probabilities are positive and sum to 1
        prob_flat = prob_dist.flatten()
        prob_flat = np.maximum(prob_flat, 0)
        prob_flat = prob_flat / np.sum(prob_flat)
        
        # Generate random indices based on probability distribution
        rng = check_random_state(None) # can switch to a seed for testing!!!
        indices = rng.choice(len(prob_flat), size=n_samples, p=prob_flat)
        
        # Convert flat indices to 2D coordinates
        y_indices = indices // width
        x_indices = indices % width
        
        # Map indices to actual coordinate values on the grid
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        
        x_samples = x_min + (x_max - x_min) * x_indices / (width - 1)
        y_samples = y_min + (y_max - y_min) * y_indices / (height - 1)
        
        return x_samples, y_samples
    
    @staticmethod
    def normalize_field(field, target_range=(0, 1)): # UNUSED
        """
        Normalize a field to a target range.
        
        Parameters:
        ----------
        field : numpy.ndarray
            Field to normalize
        target_range : tuple, optional
            Target range for normalization (default: (0, 1))
        
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


class SDFRectangle(ProbabilityDistribution):
    """
    A probability distribution defined by a rectangle's signed distance field (SDF).
    """
    
    def __init__(self, center_x, center_y, width, height, angle=0.0, 
                 smoothing=0.0, mesh_grid=None, grid_size=100, plot_range=(-5, 5)):
        """
        Initialize a rectangle SDF probability distribution.
        
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
        angle : float, optional
            Rotation angle in radians (default: 0.0)
        smoothing : float, optional
            Sigma parameter for Gaussian smoothing (default: 0.0)
        mesh_grid : tuple, optional
            Tuple containing (X, Y) meshgrid arrays. If None, will create a new meshgrid.
        grid_size : int, optional
            Resolution of the grid if creating a new meshgrid (default: 100)
        plot_range : tuple, optional
            Range of x and y coordinates if creating a new meshgrid (default: (-5, 5))
        """
        super().__init__(mesh_grid, grid_size, plot_range)
        
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.angle = angle
        self.smoothing = smoothing
        
        # Cache the SDF and probability
        self._sdf = None
        self._probability = None
    
    def get_sdf(self):
        """
        Calculate and return the signed distance field for the rectangle.
        
        Returns:
        -------
        numpy.ndarray
            SDF array
        """
        if self._sdf is None:
            self._compute_sdf()
        return self._sdf
    
    def get_probability(self):
        """
        Convert SDF to a probability-like distribution and return it.
        
        Returns:
        -------
        numpy.ndarray
            Probability distribution array
        """
        if self._probability is None:
            sdf = self.get_sdf()
            # Convert SDF to probability-like (higher inside, lower outside)
            prob = -sdf  # Invert so inside is positive
            prob = (prob + 1) / 2  # Scale to [0, 1]
            self._probability = prob
        
        return self._probability
    
    def _compute_sdf(self):
        """
        Compute the signed distance field for the rectangle.
        """
        X, Y = self.mesh_grid

        # print(np.min(X))
        
        # If there's an angle, rotate the coordinate system
        if self.angle != 0:
            # Translate to origin
            X_centered = X - self.center_x
            Y_centered = Y - self.center_y
            
            # Rotate coordinates
            cos_angle = np.cos(self.angle)
            sin_angle = np.sin(self.angle)
            X_rot = X_centered * cos_angle - Y_centered * sin_angle
            Y_rot = X_centered * sin_angle + Y_centered * cos_angle
            
            # Translate back
            X_final = X_rot + self.center_x
            Y_final = Y_rot + self.center_y
            
            # For SDF calculation, we keep the rectangle aligned with axes,
            # and instead transform the coordinate system
            X_for_sdf = X_final
            Y_for_sdf = Y_final
        else:
            X_for_sdf = X
            Y_for_sdf = Y
        
        # Calculate half dimensions
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Calculate the distance to the rectangle for each point
        # For points outside the rectangle
        dx = np.maximum(np.abs(X_for_sdf - self.center_x) - half_width, 0)
        dy = np.maximum(np.abs(Y_for_sdf - self.center_y) - half_height, 0)
        outside_distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate the inside distance (negative values)
        inside_x = half_width - np.abs(X_for_sdf - self.center_x)
        inside_y = half_height - np.abs(Y_for_sdf - self.center_y)
        inside_distance = -np.minimum(inside_x, inside_y)
        inside_distance[inside_x < 0] = 0
        inside_distance[inside_y < 0] = 0
        
        # Combine inside and outside distances to get the SDF
        sdf = np.where((np.abs(X_for_sdf - self.center_x) <= half_width) & 
                      (np.abs(Y_for_sdf - self.center_y) <= half_height),
                      inside_distance, outside_distance)
        
        # Apply smoothing if requested
        if self.smoothing > 0:
            sdf = gaussian_filter(sdf, sigma=self.smoothing)
        
        # Truncate values
        sdf = np.clip(sdf, -1, 1)
        
        self._sdf = sdf
        return sdf
    
    def get_params(self): # UNUSED, but possibly useful
        """
        Return the parameters of the rectangle.
        
        Returns:
        -------
        tuple
            (center_x, center_y, width, height, angle, smoothing)
        """
        return (self.center_x, self.center_y, self.width, self.height, 
                self.angle, self.smoothing)


class GaussianDistribution(ProbabilityDistribution):
    """
    A probability distribution defined by a 2D Gaussian.
    """
    
    def __init__(self, mean_x, mean_y, sigma_x, sigma_y, rho=0.0, weight=1.0,
                 mesh_grid=None, grid_size=100, plot_range=(-5, 5)):
        """
        Initialize a Gaussian probability distribution.
        
        Parameters:
        ----------
        mean_x : float
            Mean (center) x-coordinate
        mean_y : float
            Mean (center) y-coordinate
        sigma_x : float
            Standard deviation in x direction
        sigma_y : float
            Standard deviation in y direction
        rho : float, optional
            Correlation coefficient between x and y (-1 to 1) (default: 0.0)
        weight : float, optional
            Weight of this Gaussian in a mixture (default: 1.0)
        mesh_grid : tuple, optional
            Tuple containing (X, Y) meshgrid arrays. If None, will create a new meshgrid.
        grid_size : int, optional
            Resolution of the grid if creating a new meshgrid (default: 100)
        plot_range : tuple, optional
            Range of x and y coordinates if creating a new meshgrid (default: (-5, 5))
        """
        super().__init__(mesh_grid, grid_size, plot_range)
        
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho
        self.weight = weight
        
        # Cache the probability
        self._probability = None
    
    def get_probability(self):
        """
        Calculate and return the Gaussian probability density.
        
        Returns:
        -------
        numpy.ndarray
            Probability density array
        """
        if self._probability is None:
            X, Y = self.mesh_grid
            pos = np.dstack((X, Y))
            
            # Create the covariance matrix
            cov_matrix = np.array([
                [self.sigma_x**2, self.rho * self.sigma_x * self.sigma_y],
                [self.rho * self.sigma_x * self.sigma_y, self.sigma_y**2]
            ])
            
            # Create the multivariate normal distribution
            mean = np.array([self.mean_x, self.mean_y])
            rv = multivariate_normal(mean, cov_matrix)
            
            # Evaluate the PDF on the grid
            gaussian = rv.pdf(pos) * self.weight
            
            # Normalize to [0, 1]
            if np.max(gaussian) > 0:
                gaussian = gaussian / np.max(gaussian)
            
            self._probability = gaussian
        
        return self._probability
    
    def get_params(self):
        """
        Return the parameters of the Gaussian.
        
        Returns:
        -------
        tuple
            (mean_x, mean_y, sigma_x, sigma_y, rho, weight)
        """
        return (self.mean_x, self.mean_y, self.sigma_x, self.sigma_y, 
                self.rho, self.weight)


class GaussianMixtureDistribution(ProbabilityDistribution):
    """
    A probability distribution defined by a mixture of Gaussians.
    """
    
    def __init__(self, gaussian_params=None, mesh_grid=None, grid_size=100, plot_range=(-5, 5)):
        """
        Initialize a Gaussian mixture probability distribution.
        
        Parameters:
        ----------
        gaussian_params : list of tuples, optional
            List of (mean_x, mean_y, sigma_x, sigma_y, rho, weight) tuples for Gaussians
        mesh_grid : tuple, optional
            Tuple containing (X, Y) meshgrid arrays. If None, will create a new meshgrid.
        grid_size : int, optional
            Resolution of the grid if creating a new meshgrid (default: 100)
        plot_range : tuple, optional
            Range of x and y coordinates if creating a new meshgrid (default: (-5, 5))
        """
        super().__init__(mesh_grid, grid_size, plot_range)
        
        self.gaussians = []
        if gaussian_params:
            for params in gaussian_params:
                mean_x, mean_y, sigma_x, sigma_y, rho, weight = params
                self.add_gaussian(mean_x, mean_y, sigma_x, sigma_y, rho, weight)
        
        # The fitted GMM model from sklearn
        self.gmm = None
    
    def add_gaussian(self, mean_x, mean_y, sigma_x, sigma_y, rho=0.0, weight=1.0):
        """
        Add a Gaussian component to the mixture.
        
        Parameters:
        ----------
        mean_x : float
            Mean (center) x-coordinate
        mean_y : float
            Mean (center) y-coordinate
        sigma_x : float
            Standard deviation in x direction
        sigma_y : float
            Standard deviation in y direction
        rho : float, optional
            Correlation coefficient between x and y (-1 to 1) (default: 0.0)
        weight : float, optional
            Weight of this Gaussian in a mixture (default: 1.0)
        """
        gaussian = GaussianDistribution(
            mean_x, mean_y, sigma_x, sigma_y, rho, weight,
            mesh_grid=self.mesh_grid
        )
        self.gaussians.append(gaussian)
        self._probability = None  # Clear cache
    
    def get_probability(self):
        """
        Calculate and return the Gaussian mixture probability density.
        
        Returns:
        -------
        numpy.ndarray
            Probability density array
        """
        if self._probability is None and self.gaussians:
            # Combine all Gaussians
            combined = np.zeros_like(self.mesh_grid[0])
            
            for gaussian in self.gaussians:
                prob = gaussian.get_probability() * gaussian.weight
                combined += prob
            
            # Normalize to [0, 1]
            if np.max(combined) > 0:
                combined = combined / np.max(combined)
            
            self._probability = combined
        
        return self._probability
    
    def get_gaussian_params(self):
        """
        Return the parameters of all Gaussians in the mixture.
        
        Returns:
        -------
        list of tuples
            List of (mean_x, mean_y, sigma_x, sigma_y, rho, weight) tuples
        """
        return [gaussian.get_params() for gaussian in self.gaussians]
    
    def from_sklearn_gmm(self, gmm):
        """
        Initialize this distribution from a fitted sklearn GMM model.
        
        Parameters:
        ----------
        gmm : sklearn.mixture.GaussianMixture
            Fitted GMM model
        """
        self.gaussians = []
        self.gmm = gmm
        
        # Extract parameters from GMM
        for i in range(gmm.n_components):
            mean_x, mean_y = gmm.means_[i]
            covariance = gmm.covariances_[i]
            
            # Extract variance and correlation
            sigma_x = np.sqrt(covariance[0, 0])
            sigma_y = np.sqrt(covariance[1, 1])
            rho = covariance[0, 1] / (sigma_x * sigma_y) if sigma_x > 0 and sigma_y > 0 else 0
            
            # Add weight from GMM
            weight = gmm.weights_[i]
            
            self.add_gaussian(mean_x, mean_y, sigma_x, sigma_y, rho, weight)
        
        self._probability = None  # Clear cache


class CombinedDistribution(ProbabilityDistribution): # Do I need this at all??? This is a class that literally is just a big plus sign
    """
    A probability distribution created by combining multiple other distributions.
    """
    
    def __init__(self, distributions=None, method='sum', mesh_grid=None, 
                 grid_size=100, plot_range=(-5, 5)):
        """
        Initialize a combined probability distribution.
        
        Parameters:
        ----------
        distributions : list of ProbabilityDistribution, optional
            List of probability distributions to combine
        method : str, optional
            Method for combining distributions:
            - 'sum': Add distributions (union-like behavior)
            - 'product': Multiply distributions (intersection-like behavior)
            - 'max': Take element-wise maximum (union)
            - 'min': Take element-wise minimum (intersection)
        mesh_grid : tuple, optional
            Tuple containing (X, Y) meshgrid arrays. If None, will create a new meshgrid.
        grid_size : int, optional
            Resolution of the grid if creating a new meshgrid (default: 100)
        plot_range : tuple, optional
            Range of x and y coordinates if creating a new meshgrid (default: (-5, 5))
        """
        super().__init__(mesh_grid, grid_size, plot_range)
        
        self.distributions = distributions if distributions else []
        self.method = method
    
    # def add_distribution(self, distribution):
    #     """
    #     Add a distribution to the combination.
        
    #     Parameters:
    #     ----------
    #     distribution : ProbabilityDistribution
    #         Distribution to add
    #     """
    #     self.distributions.append(distribution)
    #     self._probability = None  # Clear cache
    
    def get_probability(self):
        """
        Calculate and return the combined probability density.
        
        Returns:
        -------
        numpy.ndarray
            Combined probability density array
        """
        if self._probability is None and self.distributions:
            # Get all distributions as arrays
            dist_arrays = [dist.get_probability() for dist in self.distributions]
            
            # Combine according to the method
            if self.method == 'sum':
                combined = np.sum(dist_arrays, axis=0)
            elif self.method == 'product':
                combined = np.ones_like(self.mesh_grid[0])
                for dist in dist_arrays:
                    combined *= dist
            elif self.method == 'max':
                combined = np.maximum.reduce(dist_arrays)
            elif self.method == 'min':
                combined = np.minimum.reduce(dist_arrays)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Normalize to [0, 1]
            if np.max(combined) > 0:
                combined = combined / np.max(combined)
            
            self._probability = combined
        
        return self._probability


class GaussianFitter:
    """
    A class for fitting Gaussian Mixture Models to probability distributions.
    """
    
    def __init__(self, n_components=10, max_iter=100):
        """
        Initialize a GaussianFitter.
        
        Parameters:
        ----------
        n_components : int, optional
            Number of Gaussian components in the mixture (default: 10)
        max_iter : int, optional
            Maximum number of iterations for EM algorithm (default: 100)
        """
        self.n_components = n_components
        self.max_iter = max_iter
        
        # The fitted GMM model
        self.gmm = None
        self.gaussian_params = None
    
    def fit(self, distribution, n_samples=10000, init_params='kmeans', 
            previous_gmm=None, warm_start=False):
        """
        Fit a GMM to a probability distribution.
        
        Parameters:
        ----------
        distribution : ProbabilityDistribution
            The distribution to fit
        n_samples : int, optional
            Number of points to sample from the distribution (default: 10000)
        init_params : str, optional
            Method to initialize GMM parameters (default: 'kmeans')
        previous_gmm : sklearn.mixture.GaussianMixture, optional
            Previously fitted GMM to use as starting point
        warm_start : bool, optional
            Whether to use previous_gmm as a warm start (default: False)
        
        Returns:
        -------
        tuple
            (fitted GMM, list of gaussian parameters)
        """
        # Sample points from the distribution
        x_samples, y_samples = distribution.sample_points(n_samples)
        samples = np.column_stack([x_samples, y_samples])
        
        # Initialize GMM
        if warm_start and previous_gmm is not None:
            # Copy previous GMM parameters for warm start
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                max_iter=self.max_iter,
                init_params='random'  # We'll set parameters manually
            )
            
            # I have no idea why, but these make the fitting a more expensive operation? keep these commented out
            # # Initialize with previous GMM parameters
            # self.gmm.means_init = previous_gmm.means_
            # self.gmm.weights_init = previous_gmm.weights_
            # self.gmm.precisions_init = previous_gmm.precisions_
        else:
            # Create new GMM
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                max_iter=self.max_iter,
                init_params=init_params
            )
        
        # Fit the GMM to the samples
        self.gmm.fit(samples) # takes .017 seconds on warm start

        # Extract parameters
        self.gaussian_params = []
        for i in range(self.n_components):
            mean_x, mean_y = self.gmm.means_[i]
            covariance = self.gmm.covariances_[i]
            
            # Extract variance and correlation
            sigma_x = np.sqrt(covariance[0, 0])
            sigma_y = np.sqrt(covariance[1, 1])
            rho = covariance[0, 1] / (sigma_x * sigma_y) if sigma_x > 0 and sigma_y > 0 else 0
            
            # Add weight from GMM
            weight = self.gmm.weights_[i]
            
            self.gaussian_params.append((mean_x, mean_y, sigma_x, sigma_y, rho, weight))
        
        gmm_dist = self.create_gmm_distribution()

        return gmm_dist
    
    def create_gmm_distribution(self, mesh_grid=None):
        """
        Create a GaussianMixtureDistribution from the fitted GMM.
        
        Parameters:
        ----------
        mesh_grid : tuple, optional
            Tuple containing (X, Y) meshgrid arrays
        
        Returns:
        -------
        GaussianMixtureDistribution
            Distribution representing the fitted GMM
        """
        if self.gaussian_params is None:
            raise ValueError("GMM not yet fitted")
        
        gmm_dist = GaussianMixtureDistribution(
            gaussian_params=self.gaussian_params,
            mesh_grid=mesh_grid
        )
        gmm_dist.gmm = self.gmm
        
        return gmm_dist
    
    def evaluate_on_grid(self, mesh_grid): # ONLY USED IN UNUSED CODE
        """
        Evaluate the fitted GMM on a mesh grid.
        
        Parameters:
        ----------
        mesh_grid : tuple
            Tuple containing (X, Y) meshgrid arrays
        
        Returns:
        -------
        numpy.ndarray
            GMM probability density evaluated on the grid
        """
        if self.gmm is None:
            raise ValueError("GMM not yet fitted")
        
        X, Y = mesh_grid
        xy_points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Get log probabilities from GMM
        log_probs = self.gmm.score_samples(xy_points)
        
        # Convert to probabilities and reshape to grid
        probs = np.exp(log_probs).reshape(X.shape)
        
        # Normalize to [0, 1] range
        if np.max(probs) > 0:
            probs = probs / np.max(probs)
        
        return probs
    
    def compute_approximation_error(self, original_distribution, metric='mse'): # UNUSED
        """
        Compute the approximation error between the original distribution and the GMM.
        
        Parameters:
        ----------
        original_distribution : ProbabilityDistribution
            Original distribution
        metric : str, optional
            Error metric to use ('mse', 'mae', 'max') (default: 'mse')
        
        Returns:
        -------
        float
            Error value
        """
        if self.gmm is None:
            raise ValueError("GMM not yet fitted")
        
        # Get original distribution
        original_prob = original_distribution.get_probability()
        
        # Evaluate GMM on the same grid
        gmm_prob = self.evaluate_on_grid(original_distribution.mesh_grid)
        
        # Compute error
        error = np.abs(original_prob - gmm_prob)
        
        if metric == 'mse':
            return np.mean(error**2)
        elif metric == 'mae':
            return np.mean(error)
        elif metric == 'max':
            return np.max(error)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class ProbabilityViewer:
    """
    A class for visualizing probability distributions.
    """
    
    def __init__(self, mesh_grid = None, grid_size=100, plot_range=(-5, 5), cmap=cm.viridis):
        """
        Initialize a ProbabilityViewer.
        
        Parameters:
        ----------
        grid_size : int, optional
            Resolution of the grid (default: 100)
        plot_range : tuple, optional
            Range of x and y coordinates to compute (default: (-5, 5))
        cmap : matplotlib.colors.Colormap, optional
            Colormap for plots (default: viridis)
        """
        self.grid_size = grid_size
        self.plot_range = plot_range
        self.cmap = cmap
        self.mesh_grid = mesh_grid

        self.ax1 = None
        self.ax2 = None

    def plot_distribution(self, distribution, show_3d=True, title=None):
        """
        wow. look. the name is literally all you need to know
        """
        

        field = distribution.get_probability()
        title = title if title else "unnamed distribution"

        # Create figure
        if show_3d:
            fig = plt.figure(figsize=(14, 6))
            self.ax1 = fig.add_subplot(1, 2, 1)
        else:
            fig, self.ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot the field as a contour plot
        X, Y = self.mesh_grid
        contour = self.ax1.contourf(X, Y, field, 50, cmap=self.cmap)
        if title:
            self.ax1.set_title(title)
        else:
            self.ax1.set_title(f'Distribution: {title}')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_aspect('equal')
        fig.colorbar(contour, ax=self.ax1, label='Probability')
        
        # Draw contour lines at specific levels
        contour_levels = np.linspace(np.min(field), np.max(field), 9)
        contour_lines = self.ax1.contour(X, Y, field, 
                                levels=contour_levels,
                                colors='white', linestyles='dashed')
        self.ax1.clabel(contour_lines, inline=True, fontsize=8)
        
        # Plot the field as a 3D surface if requested
        if show_3d:
            self.ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            surf = self.ax2.plot_surface(X, Y, field, cmap=self.cmap, antialiased=True)
            if title:
                self.ax2.set_title(f'3D View of {title}')
            else:
                self.ax2.set_title(f'3D View of Distribution {title}')
            self.ax2.set_xlabel('X')
            self.ax2.set_ylabel('Y')
            self.ax2.set_zlabel('Probability')
        
        plt.tight_layout()
        return fig

    def mark_rectangles(self, rectangles): # So far, unused

        if not self.ax1:
            raise Exception("Cannot mark rectangle before plotting distribution")
        
        for rect in rectangles:
            center_x, center_y, width, height = rect
            half_width = width / 2
            half_height = height / 2
            rect_x = center_x - half_width
            rect_y = center_y - half_height
            rectangle = plt.Rectangle((rect_x, rect_y), width, height, 
                                    edgecolor='white', facecolor='none', linewidth=2)
            self.ax1.add_patch(rectangle)

    def mark_gaussians(self, gmm_distribution):
        
        if not self.ax1:
            raise Exception("Cannot mark gaussian before plotting distribution")

        gaussian_params = gmm_distribution.get_gaussian_params()
        gaussian_centers = [(g[0], g[1]) for g in gaussian_params]

        for mean in gaussian_centers:
            self.ax1.plot(mean[0], mean[1], 'ro', markersize=5)

    def plot_error(self, original, approximation, title=None):
        """
        Plot the error between two distributions.
        
        Parameters:
        ----------
        original_index : int
            Index of the original distribution
        approximation_index : int
            Index of the approximation distribution
        title : str, optional
            Custom title for the plot
        
        Returns:
        -------
        tuple
            (Figure object, error metrics dictionary)
        """

        original = original.get_probability()
        approximation = approximation.get_probability()
        
        # Calculate error
        error = np.abs(original - approximation)
        
        # Calculate error metrics
        mse = np.mean(error**2)
        mae = np.mean(error)
        max_error = np.max(error)
        
        # Plot the error
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        X, Y = self.mesh_grid
        contour = ax.contourf(X, Y, error, 50, cmap='Reds')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Error: Original Distribution vs Approximated Distribution')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        fig.colorbar(contour, ax=ax, label='Absolute Error')
        
        # Add error metrics as text
        text_str = f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax Error: {max_error:.6f}"
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Return error metrics
        metrics = {
            'mse': mse,
            'mae': mae,
            'max_error': max_error
        }
        
        return fig, metrics

def main_2d():
    # Make distributions
    mesh_grid = ProbabilityDistribution.create_mesh_grid(grid_size=50, plot_range=(-5, 5))  # Reduced grid size
    X, Y = mesh_grid

    # Setup the figure for animation
    fig, (ax_rect, ax_gmm) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize GMM fitter
    fitter = GaussianFitter(n_components=4)

    # Initialize heatmap plots using pcolormesh
    rect_field = np.zeros_like(X)
    gmm_field = np.zeros_like(X)
    
    rect_mesh = ax_rect.pcolormesh(X, Y, rect_field, cmap='viridis', shading='auto')
    gmm_mesh = ax_gmm.pcolormesh(X, Y, gmm_field, cmap='plasma', shading='auto')
    
    # Initialize scatter plot for Gaussian centers
    scatter_gaussians, = ax_gmm.plot([], [], 'ro', markersize=5)

    # Number of frames and total angle to rotate
    n_frames = 720  # 36 frames per rotation * 2 rotations
    rotations = 2  # Number of full rotations

    def update(frame):
        # Compute rotation angle
        angle = (frame / n_frames * rotations) * 360
        angle_rad = np.radians(angle)

        # Create rotated rectangle distribution
        rectangle_1 = SDFRectangle(-1, -1, 1.0, 3.0, angle_rad, 0.2, mesh_grid=mesh_grid)
        rectangle_2 = SDFRectangle(1, 1, 1.0, 3.0, -angle_rad, 0.2, mesh_grid=mesh_grid)

        combined = CombinedDistribution(distributions=[rectangle_1, rectangle_2], mesh_grid=mesh_grid)
        rect_field = combined.get_probability()

        # Update rectangle heatmap properly
        rect_mesh.set_array(rect_field.ravel())
        rect_mesh.set_clim(vmin=np.min(rect_field), vmax=np.max(rect_field))
        ax_rect.set_title(f"Rectangle Distribution (Angle: {angle:.1f}°)")

        # Fit GMM
        start = time.time()
        fitter.fit(combined, n_samples=500)  # Reduced sample count
        gmm_distribution = fitter.create_gmm_distribution(mesh_grid=mesh_grid)
        gmm_field = gmm_distribution.get_probability()
        print(f"GMM fit time: {time.time() - start:.4f} sec")

        # Update GMM heatmap properly
        gmm_mesh.set_array(gmm_field.ravel())
        gmm_mesh.set_clim(vmin=np.min(gmm_field), vmax=np.max(gmm_field))
        ax_gmm.set_title(f"GMM Approximation (Angle: {angle:.1f}°)")

        # Update Gaussian scatter points
        gaussian_params = gmm_distribution.get_gaussian_params()
        scatter_gaussians.set_data([g[0] for g in gaussian_params], [g[1] for g in gaussian_params])

        return rect_mesh, gmm_mesh, scatter_gaussians

    # Create animation with blitting
    ani = FuncAnimation(fig, update, frames=n_frames, interval=0, blit=False, repeat=False)

    plt.tight_layout()
    plt.show()

def main_3d():
    # make distributions
    mesh_grid = ProbabilityDistribution.create_mesh_grid(grid_size=100, plot_range=(-5, 5))
    
    # Setup the figure for animation with two 3D subplots
    fig = plt.figure(figsize=(12, 6))
    ax_rect = fig.add_subplot(1, 2, 1, projection='3d')
    ax_gmm = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Initialize GMM fitter outside the animation loop
    fitter = GaussianFitter(n_components=10)
    
    # Variables to store the most recent GMM distribution
    gmm_distribution = None
    gmm_field = None
    X, Y = mesh_grid
    
    # Number of frames and total angle to rotate
    n_frames = 72  # 36 frames per rotation * 2 rotations
    rotations = 2   # Number of full rotations
    
    def update(frame):
        nonlocal gmm_distribution, gmm_field
        
        # Clear previous plots
        ax_rect.clear()
        ax_gmm.clear()
        
        # Calculate angle for current frame (2 full rotations)
        angle = (frame / n_frames * rotations) * 360
        angle_rad = np.radians(angle)
        
        # Create rectangle with current angle
        rectangle = SDFRectangle(0, 0, 1.0, 3.0, angle_rad, 0.2, mesh_grid=mesh_grid)
        combined = CombinedDistribution(distributions=[rectangle], mesh_grid=mesh_grid)
        
        # Get probability field for rectangle
        rect_field = combined.get_probability()
        
        # Update 3D surface plot for rectangle
        surf_rect = ax_rect.plot_surface(X, Y, rect_field, cmap='viridis', antialiased=True)
        ax_rect.set_title(f"Rectangle Distribution (Angle: {angle:.1f}°)")
        ax_rect.set_xlabel('X')
        ax_rect.set_ylabel('Y')
        ax_rect.set_zlabel('Probability')
        
        # Fit GMM in real-time (not too often)
        if frame % 6 == 0 or gmm_field is None:  # Only fit occasionally to speed up animation
            fitter.fit(combined, n_samples=1000)
            gmm_distribution = fitter.create_gmm_distribution(mesh_grid=mesh_grid)
            gmm_field = gmm_distribution.get_probability()
        
        # Update 3D surface plot for GMM
        surf_gmm = ax_gmm.plot_surface(X, Y, gmm_field, cmap='plasma', antialiased=True)
        ax_gmm.set_title(f"GMM Approximation (Angle: {angle:.1f}°)")
        ax_gmm.set_xlabel('X')
        ax_gmm.set_ylabel('Y')
        ax_gmm.set_zlabel('Probability')
        
        # Set consistent view angles for both plots
        ax_rect.view_init(elev=30, azim=angle)
        ax_gmm.view_init(elev=30, azim=angle)
        
        return surf_rect, surf_gmm
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False, repeat=False)
    
    # Define a function to handle animation completion
    def on_animation_finished(event):
        if event.source is ani:
            plt.close(fig)
    
    # Add the event handler
    # Note: This only works if you're using a supported backend that handles animation_finished events
    fig.canvas.mpl_connect('close_event', on_animation_finished)
    
    # Another approach is to use a timer to close after expected duration
    import threading
    timer = threading.Timer((n_frames * 200/1000) + 1, plt.close, args=[fig])
    timer.start()
    
    plt.tight_layout()
    plt.show()  

def main_simple():

    # make distributions
    mesh_grid = ProbabilityDistribution.create_mesh_grid(grid_size=100, plot_range=(-5,5))
    rectangle_example = SDFRectangle(0, 0, 1.0, 2.0, 0.0, 0.2, mesh_grid=mesh_grid)
    gaussian_example = GaussianDistribution(1,1,1,2,0,1,mesh_grid=mesh_grid)
    distributions = [rectangle_example, gaussian_example]
    combined = CombinedDistribution(distributions=distributions, mesh_grid=mesh_grid)

    # fit gmm
    fitter = GaussianFitter(n_components=10)
    import time
    start = time.time()
    fitter.fit(combined, n_samples=1000)
    print(f"initial cold start {time.time() - start}")
    gmm_distribution = fitter.create_gmm_distribution(mesh_grid=mesh_grid)




    # fit the same thing again because timing it is fun
    # start = time.time()
    # for i in range(10000):
    #     rectangle_example = SDFRectangle(0, 0, 1.0*i/100 +.1, 2.0, 0.0, 0.2, mesh_grid=mesh_grid)
    #     gaussian_example = GaussianDistribution(i/100,-i/100,1,2,0,1,mesh_grid=mesh_grid)
    #     distributions = [rectangle_example, gaussian_example]
    #     combined = CombinedDistribution(distributions=distributions, mesh_grid=mesh_grid)
    #     fitter.fit(distribution=combined, n_samples=1000, previous_gmm=gmm_distribution.gmm, warm_start=True)
    # print(f"warm start {(time.time() - start)/10000}")

    # Realistic timing:
    # 10000 trials, moving rect and gaussian
    # 0.007102960968017578 sec / trial on average

    # assume 6x speed up with parallelization. thats 884 fps.
    # Thats about 355 seconds of just doing GMM for 500 sims of 20 seconds each at 30 fps
    # 10 FPS is only 2 minutes, which is very reasonable!

    # In the rotating rectangle, this slows down to about .025 sec / trial
    # this is 3.5 times slower. around 40hz update rate



    # look at it
    viewer = ProbabilityViewer(mesh_grid=mesh_grid)
    viewer.plot_distribution(combined, title="initial")
    viewer.plot_distribution(gmm_distribution, title="GMM")

    # where art thou centers
    viewer.mark_gaussians(gmm_distribution)





    # Plot the error between original and GMM
    fig3, metrics = viewer.plot_error(combined, gmm_distribution, title="Approximation Error")
    
    # Print the learned Gaussian parameters
    print("\nLearned Gaussian Parameters:")
    print("%-8s %-8s %-8s %-8s %-8s %-8s" % ("Mean X", "Mean Y", "Sigma X", "Sigma Y", "Rho", "Weight"))
    print("-" * 60)
    
    gaussian_params = gmm_distribution.get_gaussian_params()
    for i, params in enumerate(gaussian_params):
        mean_x, mean_y, sigma_x, sigma_y, rho, weight = params
        print("%-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f" % 
              (mean_x, mean_y, sigma_x, sigma_y, rho, weight))
    
    # Print error metrics
    print("\nApproximation Error Metrics:")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"Maximum Error: {metrics['max_error']:.6f}")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":

    main_2d()