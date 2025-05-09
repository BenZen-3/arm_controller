import numpy as np

class Diffuser:

    def __init__(self, gmm_estimate_history, n_diffusion_steps=20, schedule_fn=None):
        self.gmm_estimate_history = gmm_estimate_history
        self.n_diffusion_steps = n_diffusion_steps
        self.diffused_gmm_history = []
        self.noise_history = []  # New: store noise applied

        # # Default linear schedule if none provided
        # if schedule_fn is None:
        #     self.schedule_fn = lambda t, total: 1.0 - (t / total)
        # else:
        #     self.schedule_fn = schedule_fn

        # Default linear beta schedule
        if schedule_fn is None:
            self.schedule_fn = lambda t: 1e-4 + (t / self.n_diffusion_steps) * (0.02 - 1e-4)
        else:
            self.schedule_fn = schedule_fn



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
                    sigma_scale = .1
                    rho_scale = 0
                    weight_scale = 0

                    # Apply noise cumulatively
                    new_mean_x = sqrt_1m_beta * mean_x + sqrt_beta * noise[0] * mean_scale
                    new_mean_y = sqrt_1m_beta * mean_y + sqrt_beta * noise[1] * mean_scale
                    new_sigma_x = max(sqrt_1m_beta * sigma_x + sqrt_beta * noise[2] * sigma_scale, 1e-6)
                    new_sigma_y = max(sqrt_1m_beta * sigma_y + sqrt_beta * noise[3] * sigma_scale, 1e-6)
                    new_rho = np.clip(sqrt_1m_beta * rho + sqrt_beta * noise[4] * rho_scale, -1.0, 1.0)
                    new_weight = np.clip(sqrt_1m_beta * weight + sqrt_beta * noise[5] * weight_scale, 0.0, 1.0)

                    new_g = [new_mean_x, new_mean_y, new_sigma_x, new_sigma_y, new_rho, new_weight]
                    noise_g = [noise[0] * 0.25, noise[1] * 0.25, noise[2] * 0.1, noise[3] * 0.1, noise[4] * 0.05, noise[5] * 0.1]

                    step_diffused.append(tuple(new_g))
                    step_noise.append(tuple(noise_g))

                current_gaussians = [list(g) for g in step_diffused]
                frame_diffused.insert(0, step_diffused)
                frame_noise.insert(0, step_noise)

            self.diffused_gmm_history.append(frame_diffused)
            self.noise_history.append(frame_noise)

        return self.diffused_gmm_history, self.noise_history

    # def forward_diffusion(self):
    #     """returns diffused history, noise"""

    #     self.diffused_gmm_history = []
    #     self.noise_history = []

    #     for state in self.gmm_estimate_history:
    #         frame_diffused = []
    #         frame_noise = []


    #         for step in range(self.n_diffusion_steps):

    #             diffused_states = []
    #             noise_states = []

    #             for gaussian in state:
    #                 mean_x, mean_y, sigma_x, sigma_y, rho, weight = gaussian

    #                 noise_scale = self.schedule_fn(step, self.n_diffusion_steps)
    #                 # noise_scale = 0

    #                 # Generate noise for each component
    #                 noise_mean_x = np.random.normal(0, noise_scale)* .25
    #                 noise_mean_y = np.random.normal(0, noise_scale)* .25
    #                 noise_sigma_x = np.random.normal(0, noise_scale)* .1
    #                 noise_sigma_y = np.random.normal(0, noise_scale)* .1
    #                 noise_rho = np.random.lognormal(0, noise_scale)
    #                 noise_weight = np.random.normal(0, noise_scale) * .1

    #                 # noise_mean_x = 0.1*noise_scale
    #                 # noise_mean_y = 0.1*noise_scale
    #                 # noise_sigma_x = .1*noise_scale
    #                 # noise_sigma_y = .1*noise_scale
    #                 noise_rho = 0.1*noise_scale
    #                 # noise_weight = 0.1*noise_scale

    #                 # Apply noise to get diffused values
    #                 diffused_state = (
    #                     mean_x + noise_mean_x,
    #                     mean_y + noise_mean_y,
    #                     max(sigma_x + noise_sigma_x, 1e-6),
    #                     max(sigma_y + noise_sigma_y, 1e-6),
    #                     np.clip(rho + noise_rho, -1.0, 1.0),
    #                     np.clip(weight + noise_weight, 0.0, 1.0)
    #                 )

    #                 noise_state = (
    #                     noise_mean_x,
    #                     noise_mean_y,
    #                     noise_sigma_x,
    #                     noise_sigma_y,
    #                     noise_rho,
    #                     noise_weight
    #                 )

    #                 diffused_states.append(diffused_state)
    #                 noise_states.append(noise_state)

    #             frame_diffused.append(diffused_states)
    #             frame_noise.append(noise_states)

    #         self.diffused_gmm_history.append(frame_diffused)
    #         self.noise_history.append(frame_noise)

    #     return self.diffused_gmm_history, self.noise_history

    def get_diffused_history(self):
        return self.diffused_gmm_history

    def get_noise_history(self):
        return self.noise_history
