from arm_controller.core.message_bus import MessageBus
from arm_controller.core.subscriber import Subscriber
from arm_controller.core.publisher import Publisher
from arm_controller.core.message_types import Message, NumberMessage, ListMessage, SimStateMessage, JointTorqueMessage, SimTimerMessage, JointStateMessage, CartesianMessage

from arm_controller.simulation.arm_dynamics import Arm
from arm_controller.simulation.arm_controller import NoController

import numpy as np


class SimManager:
    """Class for managing multiple simulations"""

    def __init__(self, num_sims, sim_time):
        """
        Initialize the SimManager.
        
        Parameters:
        - num_sims: Number of simulations to run in parallel.
        - sim_time: Duration of each simulation.
        """

        self.num_sims = num_sims
        self.sim_time = sim_time

    def batch_process(self):

        pass


class Simulation:
    """Class for running a single simulation"""

    def __init__(self, message_bus: MessageBus, total_time: float, frequency: int):

        self.bus = message_bus
        self.total_time = total_time
        self.frequency = frequency

        # all the simulation publishing stuffs
        self.sim_timer = Publisher(self.bus, "sim/sim_timer")
        sim_state_msg = SimStateMessage(self.frequency, self.total_time, False)
        self.bus.set_state("sim/sim_state", sim_state_msg)

        # should load the params from something like a yaml instead
        self.arm = Arm(self.bus)
        self.controller = NoController(self.bus)

        # cascade is: sim sets goal_state -> controller(joint_state, goal_state) -> arm_dynamics(torques, dt) -> arm_dynamics sets arm_state

        # better cascade: allows controller to run slower than sim speed
        # sim sets goal_state. sim sets tick -> 
        # arm checks for posted controller torques arm_dynamics(torques, dt) -> 
        # controller 

    def run(self):

        num_ticks = self.total_time // self.frequency
        dt = 1/self.frequency
        self.set_sim_state_running(True)

        for n_tick in range(num_ticks):

            # simulation runs timing. Timing deals with when goals come up
            self.bus.set_state("arm/goal_state", CartesianMessage(np.zeros(2)))

            # send update to controller
            current_time = n_tick * self.frequency
            self.sim_timer.publish(SimTimerMessage(current_time, dt))

        self.set_sim_state_running(False)

    def set_sim_state_running(self, running):

        # update state message
        current_state = self.bus.get_state("sim/sim_state")
        current_state.running = running
        self.bus.set_state("sim/sim_state", current_state)


# class Simulation:
#     """Class for running a single simulation"""

#     def __init__(self, message_bus: MessageBus):

#         # all the simulation publishing stuffs
#         self.bus = message_bus        
#         self.joint_angles_pub = Publisher(self.bus, "joint_angles")
#         self.joint_torques_pub = Publisher(self.bus, "joint_torques")
#         self.sim_time_pub = Publisher(self.bus, "sim_time")
#         self.sim_details_pub = Publisher(self.bus, "sim_details") # static publish, need to change to sticky/service soon

#         # should load the params from something like a yaml instead
#         self.arm = Arm(self.bus)
#         self.controller = Controller(self.bus)

#         # connect everyone
#         _ = Subscriber(self.bus, "joint_angles", self.controller.run_controller) # controller knows when there are new joint angles

#         # ignore below
#         # _ = Subscriber(self.bus, "joint_torques", self.arm.state_update) # arm knows when there are new torques
        
#         # NOTE: the simulation node is central to the arm node and the controller node. the simulation node will update both of them. 
#         # they do not get to talk to each other directly 


            # all the connections that I need!
            # controller.on_tick is subscribed to sim_timer
            #   controller additionally checks for any updates to the arm_goal topic
            # arm.on_control is subscribed to joint_torques


#     def run(self):

#         self.running = True

#         while(self.running):

#             # publish anlges
#             self.joint_angles_pub.publish(angles_msg)
            
#             # in return, controller publishes torques (handled by callback)

#             # update the arm state
#             self.arm.state_update(dt, self.current_control_torques)




#     def recieve_controller_torques(self, msg: JointTorqueMessage):
#         """subscribed to controller output, receive the controller torques"""
#         torques = msg
#         self.current_control_torques = torques




    # def call_me(self, message: Message):
    #     print(f"message: {message}")

    # def generate_data(self):

    #     bus = MessageBus()
    #     publisher = Publisher(bus, "my_topic")
    #     subscriber = Subscriber(bus, "my_topic", self.call_me)

    #     for i in range(100):
    #         msg = NumberMessage(1)
    #         publisher.publish(msg)