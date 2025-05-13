from abc import ABC, abstractmethod
import pickle
from pathlib import Path

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.message_types import TimingMessage, BooleanMessage, ArmStateMessage
from arm_controller.visualization.arm_visualizer import ArmVisualizer
from arm_controller.visualization.gmm_visualizer import GMMVisualizer
from arm_controller.data_synthesis.gmm_estimator import ArmFitter
from arm_controller.data_synthesis.data_diffuser import Diffuser

class Observer(ABC):

    GENERIC_NAME = "simulation"

    """collects all relevant state info and does *things* with it"""
    def __init__(self, bus: MessageBus, frequency: int = None):
        
        self.bus = bus
        self.frequency = frequency
        self.bus.subscribe("sim/observer_update", self.obsesrve_state)

        # set frequency
        if self.frequency is None:
            msg = self.bus.get_state("sim/sim_state")
            self.frequency = msg.frequency

        # get some useful states
        self.arm_description = self.bus.get_state("arm/description")
        self.sim_freq = self.bus.get_state("sim/sim_state").frequency
    
    @abstractmethod
    def obsesrve_state(self, msg: TimingMessage):
        """Observes the state on the bus and then does soemthing with it"""
        pass

    @abstractmethod
    def visualize(self):
        """visualize what the observer is seeing. Each observer has its own visualizer"""
        pass

    @classmethod
    def load(cls, path: Path):
        """
        Class method to load a recording from a file
        """
        with open(path, 'rb') as file:
            return pickle.load(file)

    def save(self):
        """pickle the observer for later"""

        save_path = self.bus.get_state("common/data_directory").path
        id = self.bus.get_state("sim/sim_state").id
        save_path = save_path.joinpath(f"{id}_{self.GENERIC_NAME}.pkl")

        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

class JointStateObserver(Observer):
    """just observes and records joint states"""
    def __init__(self, bus: MessageBus, frequency: int=None):
        super().__init__(bus, frequency)
        self.history = []

    def obsesrve_state(self, msg: TimingMessage):
        
        state = self.bus.get_state("arm/arm_state")
        self.history.append(state)

    def visualize(self, playback_speed: float=1):
        """visualize the arm's state history"""

        l_1, l_2 = self.arm_description.l_1, self.arm_description.l_2
        visualizer = ArmVisualizer(self.history, playback_speed=1, l_1=l_1, l_2=l_2, sim_rate_hz=self.frequency)
        visualizer.play()


class GMMObserver(Observer):
    """observes simulation and records the GMM equivalent"""

    def __init__(self, bus: MessageBus, frequency: int=None, num_gaussians: int=4):
        super().__init__(bus, frequency)
        self.num_gaussians = num_gaussians
        self.history = []
        self.gmm_estimate_history = [] # List of List of Tuples

        self.arm_fitter = ArmFitter(self.arm_description)

    def obsesrve_state(self, msg: TimingMessage):
        
        state: ArmStateMessage = self.bus.get_state("arm/arm_state")
        self.history.append(state)

        gmm_params = self.arm_fitter.fit_arm(state)
        self.gmm_estimate_history.append(gmm_params)

    def visualize(self, playback_speed: float=1):

        # gmm_param_history: List of frames, each a list of 6-tuples
        visualizer = GMMVisualizer(self.gmm_estimate_history, playback_speed=1.0, data_collection_hz=self.frequency)
        visualizer.play()

class DiffusionObserver(GMMObserver):
    def __init__(self, bus: MessageBus, frequency: int=None, num_gaussians: int=4, n_diffusion_steps: int=20):
        super().__init__(bus, frequency, num_gaussians)
        self.n_diffusion_steps = n_diffusion_steps
        self.bus.subscribe("sim/sim_running", self.after_sim)

        self.fused_gmm_diff_history = []
        self.fused_noise_history = []

    def after_sim(self, msg: BooleanMessage):
        """triggered after simulation ends"""
        
        diffuser = Diffuser(self.gmm_estimate_history, n_diffusion_steps=self.n_diffusion_steps, schedule_name="cosine")
        self.diffusion_history, self.noise_history, self.t_schedule_history = diffuser.forward_diffusion()
        self.fused_gmm_diff_history = self.fuse_history(self.diffusion_history)
        self.fused_noise_history = self.fuse_history(self.noise_history)
        self.fused_t_schedule_history = self.fuse_history(self.t_schedule_history)

    def fuse_history(self, history):
        """
        Flattens history[time][diffusion_step][num_gaussians][gaussian] -> history[time * diffusion_step][num_gaussians][gaussian]
        Treats each diffusion step as a unique time step in the final sequence.
        """
        fused = []

        for time_step in history:
            for diffusion_step in time_step:
                fused.append(diffusion_step)  # each is a list of Gaussians (tuples)

        return fused
    
    # def get_conditioning_history(self):
    #     """take a vector of the past history and convert it into [time * diffusion step][condition vector]"""

    #     for state in self.history:
            
    #         for step in self.n_diffusion_steps:
                
    #             condition = state.theta_1, state.theta_2
    
    def visualize(self, playback_speed: float=1):
        """visualize the gmm diffusion process"""
        
        visualizer = GMMVisualizer(self.fused_gmm_diff_history, playback_speed=1, data_collection_hz=self.frequency)
        visualizer.play()
