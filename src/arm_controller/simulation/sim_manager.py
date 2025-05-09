import multiprocessing as mp
import numpy as np
from typing import List

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.publisher import Publisher
from arm_controller.core.message_types import SimStateMessage, TimingMessage, CartesianMessage, BooleanMessage
from arm_controller.simulation.arm_dynamics import Arm
from arm_controller.simulation.arm_controller import Controller, NoController
from arm_controller.data_synthesis.sim_observer import Observer, JointStateObserver, GMMObserver, DiffusionObserver

class SimManager:
    """Class for managing multiple simulations"""

    DEFAULT_FREQUENCY = 100

    def __init__(self, bus: MessageBus, num_sims: int, total_time: float, save_sim: bool=True):
        """
        Initialize the SimManager.
        
        Parameters:
        - num_sims: Number of simulations to run in parallel.
        - sim_time: Duration of each simulation.
        """

        self.bus = bus
        self.num_sims = num_sims
        self.total_time = total_time
        self.save_sim = save_sim

    def run_single_simulation(self, sim_id: int, total_time: float, frequency: int=None) -> Observer:

        if frequency is None:
            frequency = self.DEFAULT_FREQUENCY

        # branch the global bus to keep all messages on the global bus
        sim_bus = self.bus.branch_bus()
        sim_bus.set_state("sim/sim_state", SimStateMessage(sim_id, frequency, total_time, False))
        
        # should load the params from something like a yaml instead. The order that these are created matters... sorta jank but ehhh
        theta_1, theta_2 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
        arm = Arm(sim_bus, theta_1=theta_1, theta_2=theta_2)
        controller = NoController(sim_bus)
        # observer = JointStateObserver(sim_bus)
        # observer = GMMObserver(sim_bus, 8)
        observer = DiffusionObserver(sim_bus, 10, n_diffusion_steps=40)

        # run the sim
        sim = Simulation(sim_bus, total_time, frequency, arm, controller, observer)
        sim.run()

        # save data
        if self.save_sim:
            observer.save()
        
        return observer

    def batch_process(self):
        """Run all simulations in parallel and collect the results."""

        # use all but one cpu because I enjoy using my computer
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            tasks = [(sim_id, self.total_time) for sim_id in range(self.num_sims)] # Prepare simulation arguments
            pool.starmap(self.run_single_simulation, tasks) # Run simulations in parallel


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
        self.observer_update_publisher = Publisher(self.bus, "sim/observer_update")
        self.sim_running_publisher = Publisher(self.bus, "sim/sim_running")

    def run(self):
        """
        allows dynamics, controller, recorder/observers, and goal update to run at different frequencies
        sim sets goal_state. sim publishes timing ticks ->
            arm: checks for posted controller torques. Runs arm_dynamics(torques, dt) -> arm_state
            controller: checks for posted arm_state and goal_state. Runs controller_update() -> controller_torque_state
            recorders/observers: checks for posted arm_state, goal_state, controller_torque_state. Runs what it needs to
        """

        num_ticks = self.total_time * self.sim_frequency
        dt = 1/self.sim_frequency
        self._set_sim_state_running(True)

        # different frequencies means different ticks trigger each component
        goal_update_ticks = set(self._ticks_to_run_at_freq(self.observer.frequency))
        controller_update_ticks = set(self._ticks_to_run_at_freq(self.observer.frequency))
        observer_update_ticks = set(self._ticks_to_run_at_freq(self.observer.frequency))

        for n_tick in range(num_ticks):
            timing_msg = TimingMessage(n_tick * self.sim_frequency, dt)

            self.dynamics_update_publisher.publish(timing_msg)

            if n_tick in goal_update_ticks:
                self.bus.set_state("sim/goal_state", CartesianMessage(np.zeros(2)))

            if n_tick in controller_update_ticks:
                self.controller_update_publisher.publish(timing_msg)

            if n_tick in observer_update_ticks:
                self.observer_update_publisher.publish(timing_msg)

        self._set_sim_state_running(False)
        self.sim_running_publisher.publish(BooleanMessage(False))

    def _ticks_to_run_at_freq(self, sample_freq: int) -> List[int]:
        """gives ticks to run at given freq"""

        assert(sample_freq < self.sim_frequency), f"cannot sample at higher freq than sim runs at. sample: {sample_freq}, sim: {self.sim_frequency}"
            
        num_ticks = self.total_time * self.sim_frequency
        num_sample_ticks = sample_freq * self.total_time
        return np.linspace(0, num_ticks-1, num_sample_ticks).astype(int)

    def _set_sim_state_running(self, running: bool):

        # update state message
        current_state = self.bus.get_state("sim/sim_state")
        current_state.running = running
        self.bus.set_state("sim/sim_state", current_state)
