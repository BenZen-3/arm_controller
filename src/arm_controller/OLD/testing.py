import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors

class Gaussian3D:
    """
    A class representing a 3D Gaussian distribution.
    """
    def __init__(self, mean=None, covariance=None, weight=1.0, name=None):
        """
        Initialize a 3D Gaussian distribution.
        
        Parameters:
        mean (ndarray): 3D mean vector [x, y, z]. Default is origin [0, 0, 0].
        covariance (ndarray): 3x3 covariance matrix. Default is identity matrix.
        weight (float): Weight of this Gaussian when combined with others. Default is 1.0.
        name (str): Optional name for this Gaussian.
        """
        self.mean = np.array([0.0, 0.0, 0.0]) if mean is None else np.array(mean)
        self.covariance = np.eye(3) if covariance is None else np.array(covariance)
        self.weight = weight
        self.name = name if name is not None else f"Gaussian_{id(self)}"
        
        # Compute eigendecomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)
        
        # Ensure positive eigenvalues (covariance matrix should be positive definite)
        if np.any(self.eigenvalues <= 0):
            raise ValueError("Covariance matrix must be positive definite")
    
    def pdf(self, points):
        """
        Compute the probability density function at given points.
        
        Parameters:
        points (ndarray): Array of shape (n, 3) with points [x, y, z].
        
        Returns:
        ndarray: Probability density at each point.
        """
        # Ensure points is a 2D array
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Compute difference from mean
        diff = points - self.mean
        
        # Calculate the multivariate normal PDF
        inv_cov = np.linalg.inv(self.covariance)
        exponent = np.sum(np.einsum('ij,jk,ik->i', diff, inv_cov, diff), axis=0)
        normalizer = 1.0 / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(self.covariance))
        
        return self.weight * normalizer * np.exp(-0.5 * exponent)
    
    def get_ellipsoid_points(self, confidence=2.0, n_points=30):
        """
        Generate points on the surface of the Gaussian ellipsoid.
        
        Parameters:
        confidence (float): Confidence level in standard deviations. Default is 2.0.
        n_points (int): Number of points to generate along each dimension.
        
        Returns:
        tuple: (x, y, z) coordinate arrays for the ellipsoid surface.
        """
        # Generate points on a unit sphere
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        # Reshape to get points in 3D space
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Scale by the eigenvalues (i.e., standard deviations)
        radii = np.sqrt(self.eigenvalues) * confidence
        scaled_points = points * radii
        
        # Rotate using the eigenvectors
        rotated_points = np.dot(scaled_points, self.eigenvectors.T)
        
        # Translate to mean
        transformed_points = rotated_points + self.mean
        
        # Reshape back to grid for plotting
        x_points = transformed_points[:, 0].reshape(x.shape)
        y_points = transformed_points[:, 1].reshape(y.shape)
        z_points = transformed_points[:, 2].reshape(z.shape)
        
        return x_points, y_points, z_points
    
    def __str__(self):
        return f"{self.name}: mean={self.mean}, weight={self.weight}"


class GaussianMixture3D:
    """
    A class representing a mixture of 3D Gaussian distributions.
    """
    def __init__(self, gaussians=None):
        """
        Initialize a mixture of 3D Gaussian distributions.
        
        Parameters:
        gaussians (list): List of Gaussian3D objects. Default is empty list.
        """
        self.gaussians = [] if gaussians is None else gaussians
    
    def add_gaussian(self, gaussian):
        """
        Add a Gaussian distribution to the mixture.
        
        Parameters:
        gaussian (Gaussian3D): Gaussian distribution to add.
        """
        self.gaussians.append(gaussian)
    
    def add_gaussians(self, gaussians):
        """
        Add multiple Gaussian distributions to the mixture.
        
        Parameters:
        gaussians (list): List of Gaussian3D objects to add.
        """
        self.gaussians.extend(gaussians)
    
    def pdf(self, points):
        """
        Compute the probability density function at given points.
        
        Parameters:
        points (ndarray): Array of shape (n, 3) with points [x, y, z].
        
        Returns:
        ndarray: Probability density at each point.
        """
        if not self.gaussians:
            return np.zeros(len(points))
        
        # Sum up the PDFs of all Gaussians
        total_pdf = np.zeros(len(points) if points.ndim == 2 else 1)
        for gaussian in self.gaussians:
            total_pdf += gaussian.pdf(points)
        
        return total_pdf


class Plotter3D:
    """
    A class for plotting 3D Gaussian distributions.
    """
    def __init__(self, figsize=(12, 10)):
        """
        Initialize the plotter.
        
        Parameters:
        figsize (tuple): Figure size as (width, height).
        """
        self.figsize = figsize
        self.cmap = plt.cm.viridis
        self.alpha = 0.5
    
    def plot_gaussian(self, gaussian, ax=None, confidence=2.0, color=None, alpha=None, show_axes=True):
        """
        Plot a single Gaussian ellipsoid.
        
        Parameters:
        gaussian (Gaussian3D): Gaussian distribution to plot.
        ax (Axes3D): Matplotlib 3D axis. If None, creates a new one.
        confidence (float): Confidence level in standard deviations.
        color: Color for the surface. If None, uses default colormap.
        alpha (float): Transparency (0-1). If None, uses default.
        show_axes (bool): Whether to show the principal axes.
        
        Returns:
        Axes3D: The matplotlib 3D axis.
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Get ellipsoid points
        x, y, z = gaussian.get_ellipsoid_points(confidence=confidence)
        
        # Set alpha
        alpha_value = alpha if alpha is not None else self.alpha
        
        # Create a custom colormap based on probability density if color is None
        if color is None:
            # Get distances from center for each point
            mean = gaussian.mean
            distances = np.sqrt((x - mean[0])**2 + (y - mean[1])**2 + (z - mean[2])**2)
            max_dist = np.max(distances)
            
            # Calculate color values (higher probability = lower distance = higher color value)
            color_values = 1 - distances / max_dist
            
            # Create a custom color array
            colors_array = self.cmap(color_values)
            surf = ax.plot_surface(x, y, z, facecolors=colors_array, alpha=alpha_value, linewidth=0, antialiased=True)
        else:
            surf = ax.plot_surface(x, y, z, color=color, alpha=alpha_value, linewidth=0, antialiased=True)
        
        # Plot the mean point
        ax.scatter([gaussian.mean[0]], [gaussian.mean[1]], [gaussian.mean[2]], 
                   color='red', s=50, label=f"{gaussian.name} mean")
        
        # Plot principal axes if requested
        if show_axes:
            for i in range(3):
                axis = gaussian.eigenvectors[:, i] * np.sqrt(gaussian.eigenvalues[i]) * confidence
                ax.quiver(gaussian.mean[0], gaussian.mean[1], gaussian.mean[2],
                         axis[0], axis[1], axis[2],
                         color=['r', 'g', 'b'][i],
                         arrow_length_ratio=0.1,
                         label=f"{gaussian.name} axis {i+1}")
        
        return ax
    
    def plot_mixture(self, mixture, grid_points=20, slice_plane='xy', slice_point=0,
                     plot_components=True, plot_combined=True, confidence=2.0, figsize=None):
        """
        Plot a mixture of Gaussians and their combined density.
        
        Parameters:
        mixture (GaussianMixture3D): Mixture of Gaussian distributions to plot.
        grid_points (int): Number of points along each axis for the grid.
        slice_plane (str): Plane to slice for the heatmap ('xy', 'xz', or 'yz').
        slice_point (float): Position along the third axis for the slice.
        plot_components (bool): Whether to plot individual Gaussians.
        plot_combined (bool): Whether to plot the combined density.
        confidence (float): Confidence level for ellipsoids in standard deviations.
        figsize (tuple): Figure size as (width, height). If None, uses default.
        
        Returns:
        tuple: Figure and axes.
        """
        # Use provided figsize or default
        fig_size = figsize if figsize is not None else self.figsize
        
        # Create a figure with subplots depending on what we're plotting
        if plot_components and plot_combined:
            fig = plt.figure(figsize=fig_size)
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            axes = [ax1, ax2]
        else:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111, projection='3d')
            axes = [ax]
        
        # Plot individual Gaussians if requested
        if plot_components:
            colors = plt.cm.tab10(np.linspace(0, 1, len(mixture.gaussians)))
            for i, gaussian in enumerate(mixture.gaussians):
                self.plot_gaussian(gaussian, axes[0], confidence=confidence, 
                                  color=colors[i], show_axes=False)
            
            axes[0].set_title("Individual Gaussian Components")
        
        # Plot combined density heatmap if requested
        if plot_combined:
            ax_idx = 1 if plot_components else 0
            ax_combined = axes[ax_idx]
            
            # Determine the bounds for the grid
            bounds = np.zeros((3, 2))  # [x_min, x_max], [y_min, y_max], [z_min, z_max]
            
            # Initialize with the first Gaussian's mean
            if mixture.gaussians:
                for i in range(3):
                    bounds[i, 0] = mixture.gaussians[0].mean[i] - 3 * np.sqrt(mixture.gaussians[0].covariance[i, i])
                    bounds[i, 1] = mixture.gaussians[0].mean[i] + 3 * np.sqrt(mixture.gaussians[0].covariance[i, i])
                
                # Expand bounds to include all Gaussians
                for gaussian in mixture.gaussians[1:]:
                    for i in range(3):
                        bound_low = gaussian.mean[i] - 3 * np.sqrt(gaussian.covariance[i, i])
                        bound_high = gaussian.mean[i] + 3 * np.sqrt(gaussian.covariance[i, i])
                        bounds[i, 0] = min(bounds[i, 0], bound_low)
                        bounds[i, 1] = max(bounds[i, 1], bound_high)
            else:
                # Default bounds if no Gaussians
                bounds = np.array([[-3, 3], [-3, 3], [-3, 3]])
            
            # Create grid for the specified slice plane
            if slice_plane == 'xy':
                x = np.linspace(bounds[0, 0], bounds[0, 1], grid_points)
                y = np.linspace(bounds[1, 0], bounds[1, 1], grid_points)
                X, Y = np.meshgrid(x, y)
                Z = np.ones_like(X) * slice_point
                axis_idx = 2
            elif slice_plane == 'xz':
                x = np.linspace(bounds[0, 0], bounds[0, 1], grid_points)
                z = np.linspace(bounds[2, 0], bounds[2, 1], grid_points)
                X, Z = np.meshgrid(x, z)
                Y = np.ones_like(X) * slice_point
                axis_idx = 1
            elif slice_plane == 'yz':
                y = np.linspace(bounds[1, 0], bounds[1, 1], grid_points)
                z = np.linspace(bounds[2, 0], bounds[2, 1], grid_points)
                Y, Z = np.meshgrid(y, z)
                X = np.ones_like(Y) * slice_point
                axis_idx = 0
            else:
                raise ValueError("slice_plane must be 'xy', 'xz', or 'yz'")
            
            # Create points array for PDF calculation
            points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
            
            # Calculate combined PDF at each point
            pdf_values = mixture.pdf(points).reshape(X.shape)
            
            # Normalize for visualization
            if np.max(pdf_values) > 0:
                pdf_values = pdf_values / np.max(pdf_values)
            
            # Plot the slice with heatmap
            if slice_plane == 'xy':
                surf = ax_combined.plot_surface(X, Y, Z, facecolors=self.cmap(pdf_values), alpha=0.8)
                ax_combined.set_xlabel('X')
                ax_combined.set_ylabel('Y')
                ax_combined.set_zlabel('Z')
            elif slice_plane == 'xz':
                surf = ax_combined.plot_surface(X, Y, Z, facecolors=self.cmap(pdf_values), alpha=0.8)
                ax_combined.set_xlabel('X')
                ax_combined.set_ylabel('Y')
                ax_combined.set_zlabel('Z')
            elif slice_plane == 'yz':
                surf = ax_combined.plot_surface(X, Y, Z, facecolors=self.cmap(pdf_values), alpha=0.8)
                ax_combined.set_xlabel('X')
                ax_combined.set_ylabel('Y')
                ax_combined.set_zlabel('Z')
            
            # Plot the ellipsoids in the combined plot as wireframes
            if plot_components:
                for i, gaussian in enumerate(mixture.gaussians):
                    x, y, z = gaussian.get_ellipsoid_points(confidence=confidence, n_points=20)
                    ax_combined.plot_wireframe(x, y, z, color=colors[i], alpha=0.3, linewidth=0.5)
            
            ax_combined.set_title("Combined Probability Density")
            
            # Add colorbar
            m = plt.cm.ScalarMappable(cmap=self.cmap)
            m.set_array(pdf_values)
            cbar = fig.colorbar(m, ax=ax_combined, shrink=0.5, aspect=5)
            cbar.set_label('Normalized Probability Density')
        
        # Set equal aspect ratio and limits
        for ax in axes:
            # Make a cubic bounding box
            max_range = np.max([
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                bounds[2, 1] - bounds[2, 0]
            ])
            mid_x = (bounds[0, 0] + bounds[0, 1]) / 2
            mid_y = (bounds[1, 0] + bounds[1, 1]) / 2
            mid_z = (bounds[2, 0] + bounds[2, 1]) / 2
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        return fig, axes

# Example usage
if __name__ == "__main__":
    # Create individual Gaussians
    g1 = Gaussian3D(
        mean=[0, 0, 0],
        covariance=np.eye(3),
        weight=1.0,
        name="Sphere"
    )
    
    g2 = Gaussian3D(
        mean=[2, 2, 0],
        covariance=np.array([
            [2.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.5]
        ]),
        weight=0.7,
        name="Ellipsoid"
    )
    
    g3 = Gaussian3D(
        mean=[-2, 1, -1],
        covariance=np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.4],
            [0.0, 0.4, 1.0]
        ]),
        weight=0.5,
        name="Tilted"
    )
    
    # Create a mixture
    mixture = GaussianMixture3D([g1, g2, g3])
    
    # Create a plotter and plot
    plotter = Plotter3D()
    fig, axes = plotter.plot_mixture(
        mixture,
        grid_points=40,
        slice_plane='xy',
        slice_point=0.0,
        plot_components=True,
        plot_combined=True,
        confidence=2.0
    )
    
    # Show the plot
    plt.show()