from sklearn.mixture import GaussianMixture
import numpy as np
from typing import List

from arm_controller.data_synthesis.probability import ProbabilityDistribution, SDFRectangle, GaussianMixtureDistribution, CombinedDistribution
from arm_controller.core.message_types import ArmDescriptionMessage, ArmStateMessage


class GaussianFitter:
    """
    A class for fitting Gaussian Mixture Models to probability distributions.
    """
    
    def __init__(self, n_components: int=10, max_iter: int=100):
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


class ArmFitter:

    def __init__(self, arm_description: ArmDescriptionMessage, n_gaussians: int=4, gaussian_dropoff: float=4.0):

        self.arm_description = arm_description
        self.n_gaussians = n_gaussians
        self.dropoff = gaussian_dropoff

        self.mesh_grid = ProbabilityDistribution.create_mesh_grid(grid_size=64, plot_range=(-2.1, 2.1))  # Reduced grid size
        self.fitter = GaussianFitter(n_components=n_gaussians)

    def fit_arm(self, arm_state: ArmStateMessage, n_samples: int=500) -> List:
        """
        fits the arm's sdf probability distribution to a gaussian mixture model. 
        
        returns List of (mean_x, mean_y, sigma_x, sigma_y, rho, weight) tuples
        """
        
        # create a combined distribution and fit it with gaussians
        combined_dist = self.arm_sdf(arm_state)
        gmm_dist = self.fitter.fit(combined_dist, n_samples=n_samples)
        params = gmm_dist.get_gaussian_params()

        return params

    def arm_sdf(self, arm_state: ArmStateMessage):
        """
        get that SDF baby!
        """

        l_1, l_2 = self.arm_description.l_1, self.arm_description.l_2
        x_0 = arm_state.x_0
        y_0 = arm_state.y_0
        theta_1 = arm_state.theta_1
        theta_2 = arm_state.theta_2

        width_1, width_2 = l_1, l_2
        height = .05

        # rectangle numero uno
        x1 = x_0 - l_1/2 * np.cos(theta_1)
        y1 = y_0 + l_1/2 * np.sin(theta_1)
        angle = theta_1
        lower_arm = SDFRectangle(x1, y1, width_1, height, angle, mesh_grid=self.mesh_grid, dropoff=self.dropoff)

        # rectangle numero dos
        x2 =  x_0 - l_1 * np.cos(theta_1) - l_2/2 * np.cos(theta_1 + theta_2)
        y2 = y_0 + l_1 * np.sin(theta_1) + l_2/2 * np.sin(theta_1 + theta_2)
        angle = theta_1 + theta_2
        upper_arm = SDFRectangle(x2, y2, width_2, height, angle, mesh_grid=self.mesh_grid, dropoff=self.dropoff)

        combined = CombinedDistribution(distributions=[lower_arm, upper_arm], mesh_grid=self.mesh_grid, method='max')
        return combined


