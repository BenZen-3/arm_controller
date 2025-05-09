import numpy as np

class Diffuser:

    def __init__(self, gmm_estimate_history, n_diffusion_steps=20, schedule_fn=None):
        self.gmm_estimate_history = gmm_estimate_history
        self.n_diffusion_steps = n_diffusion_steps
        self.diffused_gmm_history = []
        self.noise_history = []  # New: store noise applied

        # Default linear schedule if none provided
        if schedule_fn is None:
            self.schedule_fn = lambda t, total: 1.0 - (t / total)
        else:
            self.schedule_fn = schedule_fn

    def forward_diffusion(self):
        """returns diffused history, noise"""

        self.diffused_gmm_history = []
        self.noise_history = []

        for state in self.gmm_estimate_history:
            frame_diffused = []
            frame_noise = []

            for gaussian in state:
                mean_x, mean_y, sigma_x, sigma_y, rho, weight = gaussian

                diffused_states = []
                noise_states = []

                for step in range(self.n_diffusion_steps):
                    noise_scale = self.schedule_fn(step, self.n_diffusion_steps)

                    # Generate noise for each component
                    noise_mean_x = np.random.normal(0, noise_scale)
                    noise_mean_y = np.random.normal(0, noise_scale)
                    noise_sigma_x = np.random.normal(0, noise_scale)
                    noise_sigma_y = np.random.normal(0, noise_scale)
                    noise_rho = np.random.normal(0, noise_scale)
                    noise_weight = np.random.normal(0, noise_scale)

                    # Apply noise to get diffused values
                    diffused_state = (
                        mean_x + noise_mean_x,
                        mean_y + noise_mean_y,
                        max(sigma_x + noise_sigma_x, 1e-6),
                        max(sigma_y + noise_sigma_y, 1e-6),
                        np.clip(rho + noise_rho, -1.0, 1.0),
                        np.clip(weight + noise_weight, 0.0, 1.0)
                    )

                    noise_state = (
                        noise_mean_x,
                        noise_mean_y,
                        noise_sigma_x,
                        noise_sigma_y,
                        noise_rho,
                        noise_weight
                    )

                    diffused_states.append(diffused_state)
                    noise_states.append(noise_state)

                frame_diffused.append(diffused_states)
                frame_noise.append(noise_states)

            self.diffused_gmm_history.append(frame_diffused)
            self.noise_history.append(frame_noise)

        return self.diffused_gmm_history, self.noise_history

    def get_diffused_history(self):
        return self.diffused_gmm_history

    def get_noise_history(self):
        return self.noise_history
