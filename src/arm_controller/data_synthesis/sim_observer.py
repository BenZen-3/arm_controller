from abc import ABC, abstractmethod
import pickle

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.message_types import TimingMessage
from arm_controller.visualization.arm_plotter import ArmVisualizer

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

        # grab the arm_description for later use if needed
        self.arm_description = self.bus.get_state("arm/description")
    
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
    def __init__(self, message_bus):
        super().__init__(message_bus)
        self.history = []

    def obsesrve_state(self, msg: TimingMessage):
        
        state = self.bus.get_state("arm/arm_state")
        self.history.append(state)

    def visualize(self):
        """visualize the history"""

        # assume visualization freq is the same as the simulation freq

        l_1, l_2 = self.arm_description.l_1, self.arm_description.l_2

        plotter = ArmVisualizer(l_1, l_2)
        for state in self.history:
            plotter.plot_state(state)