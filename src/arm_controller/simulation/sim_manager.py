from arm_controller.core.message_bus import MessageBus
from arm_controller.core.subscriber import Subscriber
from arm_controller.core.publisher import Publisher
from arm_controller.core.message_types import NumberMessage, SimStateMessage, TimingMessage, CartesianMessage

from arm_controller.simulation.arm_dynamics import Arm
from arm_controller.simulation.arm_controller import Controller, NoController
from arm_controller.data_synthesis.sim_observer import JointStateObserver, Observer


import numpy as np
import multiprocessing as mp


class SimManager:
    """Class for managing multiple simulations"""

    def __init__(self, bus: MessageBus, num_sims: int, total_time: float):
        """
        Initialize the SimManager.
        
        Parameters:
        - num_sims: Number of simulations to run in parallel.
        - sim_time: Duration of each simulation.
        """

        self.bus = bus
        self.num_sims = num_sims
        self.total_time = total_time
        self.frequency = 100

    def run_single_simulation(self, sim_id, total_time, frequency):

        # branch the global bus to keep all messages on the global bus
        sim_bus = self.bus.branch_bus()
        sim_bus.set_state("sim/sim_state", SimStateMessage(sim_id, frequency, total_time, False))
        
        # should load the params from something like a yaml instead
        arm = Arm(sim_bus)
        controller = NoController(sim_bus)
        recorder = JointStateObserver(sim_bus)

        # run the sim
        sim = Simulation(sim_bus, total_time, frequency, arm, controller, recorder)
        sim.run()

        # save data
        recorder.save()



    def batch_process(self):
        """
        Run all simulations in parallel and collect the results.
        
        Returns:
        - A dictionary of {simulation_id: recording}.
        """

        results = {}
        # use all but one cpu because I enjoy using my computer
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            # Prepare simulation arguments
            tasks = [
                (sim_id, self.total_time, self.frequency) 
                for sim_id in range(self.num_sims)
            ]
            # Run simulations in parallel
            for sim_id, recording in pool.starmap(self.run_single_simulation, tasks):
                results[sim_id] = recording

        return results


class Simulation:
    """Class for running a single simulation"""

    def __init__(self, message_bus: MessageBus, total_time: float, frequency: int, arm: Arm, controller: Controller, observer: Observer=None):

        self.bus = message_bus
        self.total_time = total_time
        self.sim_frequency = frequency
        self.arm = arm
        self.controller = controller
        self.observer = observer

        # all the simulation publishing stuffs
        self.dynamics_update_publisher = Publisher(self.bus, "sim/dynamics_update")
        self.controller_update_publisher = Publisher(self.bus, "sim/controller_update")
        self.observer_update_publsiher = Publisher(self.bus, "sim/observer_update")

    def run(self):
        """
        cascade: allows dynamics, controller, recorder/observers, and goal update to run at different frequencies
        sim sets goal_state. sim publishes timing ticks ->
            arm: checks for posted controller torques. Runs arm_dynamics(torques, dt) -> arm_state
            controller: checks for posted arm_state and goal_state. Runs controller_update() -> controller_torque_state
            recorders/observers: checks for posted arm_state, goal_state, controller_torque_state. Runs what it needs to
        """

        num_ticks = self.total_time * self.sim_frequency
        dt = 1/self.sim_frequency
        self.set_sim_state_running(True)

        # these things can be at different frequenies than the actual simulation, update this later 
        goal_update = True
        controller_update = True
        observer_update = True

        for n_tick in range(num_ticks):
            current_time = n_tick * self.sim_frequency

            self.dynamics_update_publisher.publish(TimingMessage(current_time, dt))

            if goal_update:
                self.bus.set_state("sim/goal_state", CartesianMessage(np.zeros(2)))

            if controller_update:
                self.controller_update_publisher.publish(TimingMessage(current_time, dt))

            if self.observer and observer_update:
                self.observer_update_publsiher.publish(TimingMessage(current_time, dt))

        self.set_sim_state_running(False)

    def set_sim_state_running(self, running):

        # update state message
        current_state = self.bus.get_state("sim/sim_state")
        current_state.running = running
        self.bus.set_state("sim/sim_state", current_state)
