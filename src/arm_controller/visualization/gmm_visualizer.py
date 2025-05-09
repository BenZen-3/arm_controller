import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

from arm_controller.data_synthesis.probability import GaussianMixtureDistribution
from arm_controller.data_synthesis.probability import ProbabilityDistribution

# TODO: SEPARATE VIEW AND PLOT RANGES. VIEW HAS TO BE LESS THAN OR EQUAL OR PLOT

class GMMVisualizer:
    DEFAULT_PLAYBACK_HZ = 30

    def __init__(
        self,
        gmm_param_history: List[List[Tuple[float, float, float, float, float, float]]],
        playback_speed: float = 1.0,
        data_collection_hz: int = 100,
        grid_size: int = 64,
        plot_range: Tuple[float, float] = (-2.1, 2.1),
        view_range: Tuple[float, float] = (-2.1, 2.1)
    ):
        """
        gmm_param_history: list of list of Gaussian tuples over time.
        Each frame is a list of (mean_x, mean_y, sigma_x, sigma_y, rho, weight)
        """

        self.gmm_param_history = self._downsample(gmm_param_history, data_collection_hz, self.DEFAULT_PLAYBACK_HZ)
        self.interval_ms = (1000 / self.DEFAULT_PLAYBACK_HZ) / playback_speed
        self.mesh_grid = ProbabilityDistribution.create_mesh_grid(grid_size=grid_size, plot_range=plot_range)
        self.X, self.Y = self.mesh_grid

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("GMM Approximation Over Time")
        self.ax.set_xlim(view_range)
        self.ax.set_ylim(view_range)
        self.ax.set_aspect('equal')

        # Initialize heatmap and scatter
        self.gmm_field = np.zeros_like(self.X)
        self.heatmap = self.ax.pcolormesh(self.X, self.Y, self.gmm_field, cmap='plasma', shading='auto')
        self.scatter, = self.ax.plot([], [], 'ro', markersize=3)

        # hacky pause stuff. looks better this way
        self.nth_frame_pause = None
        self.pause_duration = 0

    def _downsample(self, data, sim_rate, target_rate):
        num_input_frames = len(data)
        total_time = num_input_frames / sim_rate
        num_frames = int(total_time * target_rate)
        sample_times = np.linspace(0, num_input_frames - 1, num=num_frames).astype(int)
        return [data[i] for i in sample_times]
    
    def _update(self, frame):
        gmm_params = self.gmm_param_history[frame]
        gmm_dist = GaussianMixtureDistribution(gmm_params, mesh_grid=self.mesh_grid)
        prob = gmm_dist.get_probability()

        self.heatmap.set_array(prob.ravel())
        self.heatmap.set_clim(vmin=np.min(prob), vmax=np.max(prob))

        # Scatter centers
        x = [g[0] for g in gmm_params]
        y = [g[1] for g in gmm_params]
        self.scatter.set_data(x, y)

        return self.heatmap, self.scatter

    def play(self):
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.gmm_param_history),
            interval=self.interval_ms,
            blit=False,
            repeat=False
        )
        plt.tight_layout()
        plt.show()
