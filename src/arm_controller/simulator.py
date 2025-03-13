from .arm import Arm
from . import utils
from .controller import KinematicController, ArmTrajectory
from .player import SimulationPlayer
import numpy as np
from enum import Enum
import math
from decimal import Decimal
import multiprocessing as mp
import json
import pickle

# TODO: Make the simulation freq different than the recording FPS

class VoxelState(Enum):
    """
    state of the voxel
    """
    NO_FILL = 0
    FILLED_ARM = 255


class Recording:
    frame_dtype = np.uint8

    def __init__(self, prompt=None):
        """
        Initialize a recording
        """
        self.sim_prompt = prompt
        self._frame_sequence = []
        self._arm_state_history = []
        
        # Direct attributes instead of metadata dict
        self.date = ''
        self.frame_width = None
        self.frame_height = None
        self.voxel_size = None
        self.fps = None
        self.arm_l1 = None
        self.arm_l2 = None
        self.arm_m1 = None
        self.arm_m2 = None
        self.name = None

    def init_from_file(self, path):
        """
        Initialize the recording from a pickled file
        """
        with open(path, 'rb') as file:
            loaded_data = pickle.load(file)
            
        # Restore all attributes from the loaded object
        self.__dict__.update(loaded_data.__dict__)

    def init_for_recording(self, frame_width, frame_height, voxel_size, fps, arm, name=None):
        """
        Initialize parameters for recording
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.voxel_size = voxel_size
        self.fps = fps
        
        # Store arm parameters directly
        self.arm_l1 = arm.l1
        self.arm_l2 = arm.l2
        self.arm_m1 = arm.m1
        self.arm_m2 = arm.m2
        
        # Keep a reference to the arm for real-time recording
        self._arm_reference = arm
        
        self.date = utils.pretty_date()
        self.name = name if name else f"simulation_data_{self.date}"

    def record_frame(self, voxel_grid):
        """
        Copies the voxel array and appends it to the frame sequence
        """
        self._frame_sequence.append(np.copy(voxel_grid))

    def record_angles(self, theta_1, theta_2):
        """
        Records arm state
        """
        state = (theta_1, theta_2)
        self._arm_state_history.append(state)

    def save(self, id=0, save_folder=None):
        """
        Save the object using pickle
        """
        save_path = save_folder.joinpath(f"{id}_{self.name}.pkl")
        with open(save_path, 'wb') as file:
            # Remove the _arm_reference before pickling if it exists
            # to avoid serializing the entire arm object
            if hasattr(self, '_arm_reference'):
                arm_ref = self._arm_reference
                self._arm_reference = None
                pickle.dump(self, file)
                self._arm_reference = arm_ref
            else:
                pickle.dump(self, file)

    @property
    def frame_sequence(self): 
        """
        Frame sequence getter. Always return array
        """
        return np.asarray(self._frame_sequence)
    
    @property
    def arm_state_history(self):
        """
        Arm state history getter
        """
        return self._arm_state_history

    def get_float_frame_seq(self):
        """
        Model requires float32. This stores as uint8. Output as float32
        """
        normalize_factor = np.float32(np.iinfo(self.frame_dtype).max)
        return self.frame_sequence.astype(np.float32) / normalize_factor

    @classmethod
    def frame_printer(cls, frame):
        """
        Pretty printer for a frame
        """
        for row in np.flipud(frame):
            row_output = ""
            for voxel in row:
                if round(voxel) != VoxelState.NO_FILL.value:
                    row_output += "-"
                else:
                    row_output += " "
            print(row_output)
            
    @classmethod
    def load(cls, path):
        """
        Class method to load a recording from a file
        """
        with open(path, 'rb') as file:
            return pickle.load(file)




class Simulator:

    def __init__(self, width, height, voxel_size, arm, controller=None):

        # these are in meters
        self.width = width
        self.height = height
        self.voxel_size = voxel_size

        self.arm = arm
        self.controller = controller

        # simulation init
        self.running = False
        self.external_control = False
        self.fps = 30 # TODO: make this higher... need higher accuracy than 10 hz

        self.check_conditions()
        self.check_control()
        self.generate_voxels()

        # recording
        self.recording = Recording()
        self.recording.init_for_recording(self.width, self.height, self.voxel_size, self.fps, self.arm)

    def check_conditions(self):
        """
        checks init conditions
        """

        # TODO: check that there are an odd number of voxels height and voxels width (to center the arm in a voxel)
        assert (Decimal(str(self.width)) % Decimal(str(self.voxel_size)) == Decimal('0.0')), "Voxel Size does not work with width!"
        assert (Decimal(str(self.height)) % Decimal(str(self.voxel_size)) == Decimal('0.0')), "Voxel Size does not work with height!"

    def check_control(self):
        """
        yeeeee
        """
        if self.controller:
            self.external_control = True

    def generate_voxels(self):
        """
        Generates a 2D grid of voxels based on the given width, height, and voxel size.
        """

        num_hor_vox = int(Decimal(str(self.width)) // Decimal(str(self.voxel_size)))
        num_vert_vox = int(Decimal(str(self.height)) // Decimal(str(self.voxel_size)))
        self.voxels = np.ones([num_hor_vox, num_vert_vox], dtype=Recording.frame_dtype) * VoxelState.NO_FILL.value

    def control_arm(self, dt):
        """
        control the arm
        """

        if self.external_control:
            self.controller.arm_update(dt=dt)
            self.running = not self.controller.is_complete
        else:
            self.arm.state_update(dt=dt)

    def run(self, sim_time=10):
        """
        run the sim
        """

        self.running = True
        dt = 1/self.fps
        total_time = 0
        
        while self.running:

            self.control_arm(dt)
            self.fill_arm_voxels()
            self.recording.record_frame(self.voxels)
            self.recording.record_angles(self.arm.state.theta1, self.arm.state.theta2)

            # timing, can put in while loop if nothing else breaks the total time?
            total_time += dt
            if total_time >= sim_time and not self.external_control: self.running = False

        return self.recording

    def fill_arm_voxels(self):
        """
        fill all my voxels
        """

        filled = self.check_arm_occupancy()
        self.voxels *= VoxelState.NO_FILL.value # remove all previous filled areas

        for location in filled:
            x,y = location
            self.voxels[x,y] = VoxelState.FILLED_ARM.value

    def check_arm_occupancy(self):
        """
        Checks where the arm is by casting a ray along each of the arm linkages
        """

        base, j1, j2 = self.arm.cartesian_joint_locations()
        v1 = self.raycast(self.voxels, base, j1, self.voxel_size)
        v2 = self.raycast(self.voxels, j1, j2, self.voxel_size)
        return v1 + v2
    
    def raycast(self, grid, start, end, voxel_size):
            """
            Perform a raycast on a grid of voxels and collect all collisions.
            Optimized with Numba.
            
            Parameters:
            - grid: 2D numpy array of voxel states (e.g., 1 for occupied, 0 for empty).
            - start: (x0, y0) start point of the ray.
            - end: (x1, y1) end point of the ray.
            - voxel_size: Size of each voxel.
            
            Returns:
            - List of voxel indices that the ray intersects.
            """
            
            x0, y0 = start
            x1, y1 = end

            # Convert world coordinates to grid indices
            x0_idx = int(x0 // voxel_size)
            y0_idx = int(y0 // voxel_size)

            dx = x1 - x0
            dy = y1 - y0

            step_x = 1 if dx > 0 else -1
            step_y = 1 if dy > 0 else -1

            t_max_x = ((x0_idx + (step_x > 0)) * voxel_size - x0) / dx if dx != 0 else float('inf')
            t_max_y = ((y0_idx + (step_y > 0)) * voxel_size - y0) / dy if dy != 0 else float('inf')
            t_delta_x = voxel_size / abs(dx) if dx != 0 else float('inf')
            t_delta_y = voxel_size / abs(dy) if dy != 0 else float('inf')

            vox_row = x0_idx
            vox_col = y0_idx

            max_ray_len = math.sqrt(dx**2 + dy**2)
            collided_voxels = []

            # DDA Traversal Loop
            while True:
                # Check if within bounds
                if 0 <= vox_row < grid.shape[1] and 0 <= vox_col < grid.shape[0]:
                    collided_voxels.append((vox_col, vox_row))
                else:
                    break  # Exit if the ray goes out of bounds

                # Calculate ray length and exit if it exceeds maximum
                ray_len = math.sqrt((vox_row * voxel_size - x0)**2 + (vox_col * voxel_size - y0)**2)
                if ray_len > max_ray_len:
                    break

                # Move to the next voxel
                if t_max_x < t_max_y:
                    t_max_x += t_delta_x
                    vox_row += step_x
                else:
                    t_max_y += t_delta_y
                    vox_col += step_y

            return collided_voxels


class BatchProcessor:

    def __init__(self, num_sims, sim_time):
        """
        Initialize the BatchProcessor.
        
        Parameters:
        - num_sims: Number of simulations to run in parallel.
        - sim_time: Duration of each simulation.
        """

        self.num_sims = num_sims
        self.sim_time = sim_time
        self.save_path = utils.get_data_folder()

    @staticmethod
    def run_single_simulation(sim_id, sim_time, width=4.2, height=4.2, voxel_size=0.065625, save_path = None):
        """
        Run a single simulation and return its recording.
        
        Parameters:
        - sim_id: Identifier for the simulation.
        - sim_time: Duration of the simulation.
        - width, height: Dimensions of the simulation area.
        - voxel_size: Size of each voxel in the simulation grid.

        Returns:
        - A tuple of (sim_id, recording).
        """
        t1, t2 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)

        # TODO: GET SOME RANDOMIZATIO IN THE ARM STATS
        arm = Arm(x0=width / 2, y0=height / 2, theta1=t1, theta2=t2, l1=1, l2=1, m1=1, m2=1, g=9.8) 
        sim = Simulator(width, height, voxel_size, arm)
        recording = sim.run(sim_time)
        if save_path is not None:
            recording.save(sim_id, save_path)
        return sim_id, recording

    def batch_process(self):
        """
        Run all simulations in parallel and collect the results.
        
        Returns:
        - A dictionary of {simulation_id: recording}.
        """
        results = {}
        # use all but one cpu because I enjoy ussing my computer
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            # Prepare simulation arguments
            tasks = [
                (sim_id, self.sim_time, 4.2, 4.2, 0.065625, self.save_path) 
                for sim_id in range(self.num_sims)
            ]
            # Run simulations in parallel
            for sim_id, recording in pool.starmap(BatchProcessor.run_single_simulation, tasks):
                results[sim_id] = recording

        return results









def run_controller_sim(all_json_input):

    print("running controller")

    width=4.2
    height=4.2
    voxel_size=0.065625
    t1, t2 = -np.pi/2, np.pi/2#np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)

    for json_trajectory in all_json_input:

        arm = Arm(x0=width / 2, y0=height / 2, theta1=t1, theta2=t2, l1=1, l2=1, m1=1, m2=1, g=9.8)
        arm_trajectory = ArmTrajectory(json_trajectory)
        controller = KinematicController(arm, arm_trajectory)

        sim = Simulator(width, height, voxel_size, arm, controller)
        player = SimulationPlayer(800, 800)
        player.realtime_play(sim)

def run_single_simulation(sim_id, json_trajectory, save_path, width=4.2, height=4.2, voxel_size=0.065625):
    t1, t2 = -np.pi/2, np.pi/2
    
    arm = Arm(x0=width / 2, y0=height / 2, theta1=t1, theta2=t2, l1=1, l2=1, m1=1, m2=1, g=9.8)
    arm_trajectory = ArmTrajectory(json_trajectory)
    controller = KinematicController(arm, arm_trajectory)
    sim = Simulator(width, height, voxel_size, arm, controller)
    
    recording = sim.run()
    if save_path is not None:
        recording.sim_prompt = json_trajectory['text_prompt']
        recording.save(sim_id, save_path)
    
    return sim_id, recording

def generate_sim_data(all_json_input):
    print("Running controller in parallel")
    save_path = utils.get_data_folder()
    
    results = {}
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        tasks = [(sim_id, json_trajectory, save_path) for sim_id, json_trajectory in enumerate(all_json_input)]
        for sim_id, recording in pool.starmap(run_single_simulation, tasks):
            results[sim_id] = recording
    
    return results

