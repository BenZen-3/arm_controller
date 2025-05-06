from abc import ABC, abstractmethod
import pickle

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.message_types import TimingMessage
from arm_controller.visualization.arm_visualizer import ArmVisualizer

class Observer(ABC):

    GENERIC_NAME = "simulation"

    """collects all relevant state info and does *things* with it"""
    def __init__(self, bus: MessageBus, frequency: float = None):
        
        self.bus = bus
        self.bus.subscribe("sim/observer_update", self.obsesrve_state)

        # set frequency
        if frequency is None:
            msg = self.bus.get_state("sim/sim_state")
            self.frequency = msg.frequency

        # get some useful states
        self.arm_description = self.bus.get_state("arm/description")
        self.freq = self.bus.get_state("sim/sim_state").frequency
    
    @abstractmethod
    def obsesrve_state(self, msg: TimingMessage):
        """Observes the state on the bus and then does soemthing with it"""
        pass

    @abstractmethod
    def visualize(self):
        """visualize what the observer is seeing. Each observer has its own visualizer"""
        pass

    def save(self):
        """pickle the observer for later"""

        save_path = self.bus.get_state("common/data_directory").path
        id = self.bus.get_state("sim/sim_state").id
        save_path = save_path.joinpath(f"{id}_{self.GENERIC_NAME}.pkl")

        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

class JointStateObserver(Observer):
    """just observes and records joint states"""
    def __init__(self, message_bus):
        super().__init__(message_bus)
        self.history = []

    def obsesrve_state(self, msg: TimingMessage):
        
        state = self.bus.get_state("arm/arm_state")
        self.history.append(state)

    def visualize(self):
        """visualize the arm's state history"""

        l_1, l_2 = self.arm_description.l_1, self.arm_description.l_2
        visualizer = ArmVisualizer(self.history, playback_speed=.5, l_1=l_1, l_2=l_2, sim_rate_hz=self.freq)
        visualizer.play()

class GMMObserver(Observer):
    """observes simulation and records the GMM equivalent"""

    def __init__(self, bus: MessageBus):
        super().__init__(bus)
        self.history = []

    def obsesrve_state(self, msg: TimingMessage):
        
        state = self.bus.get_state("arm/arm_state")
        self.history.append(state)

        # ESTIMATE THE ARM WITH A GMM
