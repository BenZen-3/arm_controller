import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DiffusionInput:
    diffused_gmm_history: Any
    noise_history: Any
    noise_scale_history: Any
    conditioning_history: Any


class Diffuser:
    def __init__(self, gmm_estimate_history, n_diffusion_steps: int = 20, schedule_name: str = "linear"):
        self.n_diffusion_steps = n_diffusion_steps
        self.schedule_name = schedule_name
        self.gmm_estimate_history = gmm_estimate_history  # List[Frame[List[GMM]]]
        
        # Convert input to NumPy array of shape (n_frames, n_gaussians, 6)
        self.data = np.array([
            [list(g) for g in frame]
            for frame in gmm_estimate_history
        ], dtype=np.float32)

        self.n_frames, self.n_gaussians, self.n_params = self.data.shape
        assert self.n_params == 6, "Each Gaussian must have 6 parameters"

        self._prepare_schedule()
        self.diffused_gmm_history = None
        self.noise_history = None

    def _prepare_schedule(self):
        t = np.linspace(0, 1, self.n_diffusion_steps + 1)[1:]  # Skip t=0
        if self.schedule_name == "linear":
            beta = 1e-4 + t * (0.02 - 1e-4)
        elif self.schedule_name == "cosine":
            beta = np.clip(0.5 * (1 - np.cos(np.pi * t)) * (0.02 - 1e-4), 1e-5, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_name}")
        
        alpha = 1.0 - beta
        self.alpha_bar = np.cumprod(alpha)

    def forward_diffusion(self):
        x_0 = self.data  # shape: (n_frames, n_gaussians, 6)
        T = self.n_diffusion_steps

        scales = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # scaling per param

        diffused = np.zeros((self.n_frames, T, self.n_gaussians, 6), dtype=np.float32)
        noises = np.zeros_like(diffused)
        t_schedule = np.zeros((self.n_frames, T), dtype=np.int32)

        for t in range(T):
            sqrt_alpha_bar = np.sqrt(self.alpha_bar[t])
            sqrt_1m_alpha_bar = np.sqrt(1 - self.alpha_bar[t]) # sqrt(1 - a_bar)

            # chat made this sick *x_0.shape. def stealing that
            eps = np.random.randn(*x_0.shape).astype(np.float32) * scales 

            x_t = sqrt_alpha_bar * x_0 + sqrt_1m_alpha_bar * eps
            diffused[:, t] = x_t
            noises[:, t] = eps
            t_schedule[:, t] = t

        # Clip rho and weight for numerical stability
        diffused[..., 4] = np.clip(diffused[..., 4], -1.0, 1.0)
        diffused[..., 5] = np.clip(diffused[..., 5], 0.0, 1.0)  

        # set the variables
        self.diffused_gmm_history = diffused
        self.noise_history = noises
        self.t_schedule = t_schedule

        return self.diffused_gmm_history, self.noise_history, self.t_schedule

    def get_diffused_history(self):
        return self.diffused_gmm_history

    def get_noise_history(self):
        return self.noise_history
