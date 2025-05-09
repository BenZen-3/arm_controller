import numpy as np

class Diffuser:

    def __init__(self, gmm_estimate_history, n_diffusion_steps: int=20, schedule_name: str="linear"):
        self.gmm_estimate_history = gmm_estimate_history
        self.n_diffusion_steps = n_diffusion_steps
        self.diffused_gmm_history = []
        self.noise_history = [] 

        self.set_beta_schedule(schedule_name)

    def set_beta_schedule(self, name: str="linear"):

        match name:
            case "linear":
                self.schedule_fn = lambda t: 1e-4 + (t / self.n_diffusion_steps) * (0.02 - 1e-4)
            case "cosine":
                self.schedule_fn = lambda t: min(0.999, max(1e-5, 0.5 * (1 - np.cos(np.pi * t / self.n_diffusion_steps)))) * (0.02 - 1e-4)

    def forward_diffusion(self):
        self.diffused_gmm_history = []
        self.noise_history = []

        for frame in self.gmm_estimate_history:
            current_frame = list(frame)  # start from original Gaussians
            frame_diffused = []
            frame_noise = []

            # Initialize current state per Gaussian
            current_gaussians = [list(g) for g in current_frame]  # deep copy for mutation

            for step in range(1, self.n_diffusion_steps + 1):
                beta_t = self.schedule_fn(step)
                sqrt_1m_beta = np.sqrt(1 - beta_t)
                sqrt_beta = np.sqrt(beta_t)

                step_diffused = []
                step_noise = []

                for g in current_gaussians:
                    mean_x, mean_y, sigma_x, sigma_y, rho, weight = g

                    # Sample noise for each parameter
                    noise = np.random.normal(0, 1, size=6)

                    mean_scale = 3
                    sigma_scale = 1
                    rho_scale = 1
                    weight_scale = 1

                    # Apply noise cumulatively
                    new_mean_x = sqrt_1m_beta * mean_x + sqrt_beta * noise[0] * mean_scale
                    new_mean_y = sqrt_1m_beta * mean_y + sqrt_beta * noise[1] * mean_scale
                    new_sigma_x = max(sqrt_1m_beta * sigma_x + sqrt_beta * noise[2] * sigma_scale, 1e-6)
                    new_sigma_y = max(sqrt_1m_beta * sigma_y + sqrt_beta * noise[3] * sigma_scale, 1e-6)
                    new_rho = np.clip(sqrt_1m_beta * rho + sqrt_beta * noise[4] * rho_scale, -1.0, 1.0)
                    new_weight = np.clip(sqrt_1m_beta * weight + sqrt_beta * noise[5] * weight_scale, 0.0, 1.0)

                    new_g = [new_mean_x, new_mean_y, new_sigma_x, new_sigma_y, new_rho, new_weight]
                    noise_g = [noise[0] * mean_scale, noise[1] * mean_scale, noise[2] * sigma_scale, noise[3] * sigma_scale, noise[4] * rho_scale, noise[5] * weight_scale]

                    step_diffused.append(tuple(new_g))
                    step_noise.append(tuple(noise_g))

                current_gaussians = [list(g) for g in step_diffused]
                frame_diffused.insert(0, step_diffused)
                frame_noise.insert(0, step_noise)

            self.diffused_gmm_history.append(frame_diffused)
            self.noise_history.append(frame_noise)

        return self.diffused_gmm_history, self.noise_history

    def get_diffused_history(self):
        return self.diffused_gmm_history

    def get_noise_history(self):
        return self.noise_history
