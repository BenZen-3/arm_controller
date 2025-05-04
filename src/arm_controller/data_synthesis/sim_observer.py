from abc import ABC, abstractmethod

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.message_types import TimingMessage

import pickle


class Observer(ABC):
    """collects all relevant state info and does *things* with it"""
    def __init__(self, bus: MessageBus, frequency: float = None):
        
        self.bus = bus
        self.bus.subscribe("sim/observer_update", self.obsesrve_state)

        # set frequency
        if frequency is None:
            msg = self.bus.get_state("sim/sim_state")
            self.frequency = msg.frequency
    
    @abstractmethod
    def obsesrve_state(self, msg: TimingMessage):
        """Observes the state on the bus and then does soemthing with it"""
        pass

    def save(self):
        
        save_path = self.bus.get_state("common/data_directory").path
        id = self.bus.get_state("sim/sim_state").id
        generic_name = "simulation"

        save_path = save_path.joinpath(f"{id}_{generic_name}.pkl")

        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

class JointStateObserver(Observer):
    def __init__(self, message_bus):
        super().__init__(message_bus)
        self.history = []

    def obsesrve_state(self, msg: TimingMessage):
        
        state = self.bus.get_state("arm/arm_state")
        self.history.append(state)