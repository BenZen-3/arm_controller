from scipy.stats import multivariate_normal
from sklearn.utils import check_random_state
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability-distribution-like objects.
    
    All probability distributions should be able to:
    1. Return their probability density values over a meshgrid
    2. Sample points from their distribution
    3. Be combined with other distributions
    """
    
    def __init__(self, mesh_grid=None, grid_size: int=100, plot_range: Tuple[int, int]=(-5, 5)):
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

        self.plot_range = plot_range

        if mesh_grid is None:
            self.grid_size = grid_size
            self.mesh_grid = self.create_mesh_grid(grid_size, plot_range)
        else:
            self.mesh_grid = mesh_grid
            self.grid_size = np.shape(mesh_grid)[1] # JANK
        
        # The cached probability density distribution
        self._probability = None
    
    @staticmethod
    def create_mesh_grid(grid_size: int=100, plot_range: Tuple[int, int]=(-5, 5)):
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
    
    def sample_points(self, n_samples: int=10000):
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
    def normalize_field(field, target_range=(0, 1)): # UNUSED DELETEEEEEEEEEEEEEEE
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
    
    def __init__(self, center_x: int, center_y: int, width: int, height: int, angle: float=0.0, 
                 mesh_grid=None, grid_size: int=100, plot_range: Tuple[int, int]=(-5, 5), dropoff = 1):
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
        self.dropoff = dropoff
        
        # Cache the SDF and probability
        self._sdf = None
        self._probability = None
    
    def get_sdf(self): # UNUSED. DELETE
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
            sdf = self._compute_sdf()

            # Convert SDF to probability-like (higher inside, lower outside)
            prob = -sdf  # Invert so inside is positive
            prob = (prob + 1) / 2  # Scale to [0, 1]
            self._probability = prob
        
        return self._probability
    
    def _compute_sdf(self):
        """
        Compute the signed distance field for the rectangle.
        """

        # already have sdf
        if self._sdf is not None: 
            return self._sdf

        X, Y = self.mesh_grid

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
        outside_distance = np.sqrt(dx**2 + dy**2)*self.dropoff
        
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
        
        # Truncate values
        sdf = np.clip(sdf, -1, 1)
        
        self._sdf = sdf
        return sdf
    
    def get_params(self): # UNUSED DELETE
        """
        Return the parameters of the rectangle.
        
        Returns:
        -------
        tuple
            (center_x, center_y, width, height, angle, dropoff)
        """
        return (self.center_x, self.center_y, self.width, self.height, 
                self.angle, self.dropoff)


class GaussianDistribution(ProbabilityDistribution):
    """
    A probability distribution defined by a 2D Gaussian.
    """
    
    def __init__(self, mean_x: float, mean_y: float, sigma_x: float, sigma_y: float, rho: float=0.0, weight: float=1.0,
                 mesh_grid=None, grid_size: int=100, plot_range: Tuple[int, int]=(-5, 5)):
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

            # matrix is by definition symmetric. Not always positive definite. check det.
            if np.linalg.det(cov_matrix) < 0:
                return np.zeros((self.grid_size, self.grid_size))
            
            # Create the multivariate normal distribution
            mean = np.array([self.mean_x, self.mean_y])
            try:
                rv = multivariate_normal(mean, cov_matrix, allow_singular=True)
            except Exception as e:
                print(cov_matrix)


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
        if gaussian_params is not None:
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
    
    def from_sklearn_gmm(self, gmm): # currently unused... useful for creating one?
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

