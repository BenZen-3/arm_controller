from . import Arm
import numpy as np
from enum import Enum
import math
from datetime import datetime
from decimal import Decimal
import multiprocessing as mp


class VoxelState(Enum):
    NO_FILL = 0
    FILLED_ARM = 1
    FILLED_OBSTACLE = 2


class Recording:

    output_folder = "data"

    def __init__(self):
        
        self.frame_sequence = []

    def init_from_file(self, name):
        """
        init the recording from a file
        """
        pass

    def init_for_recording(self, frame_width, frame_height, voxel_size, fps, arm, name=None):
        
        # in pixels
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.voxel_size = voxel_size

        self.fps = fps

        # arm details need to be saved
        self.arm = arm

        # TODO: this is BS
        if name == None:
            # holy mother of god this is dumb
            self.name = f"simulation_data_{datetime.now()}".replace(" ", "_").rsplit(".", 1)[0].replace(":", "_") 
        else:
            self.name = name

    def record_frame(self, voxel_grid):
        """
        copies the voxel array and appends it to the frame seq
        """
        print(np.shape(voxel_grid))
        self.frame_sequence.append(np.copy(voxel_grid))

    def save(self, id=0, entry_point=None):

        save_path = entry_point.joinpath(f"data/{id}_{self.name}")
        np.save(save_path,self.frame_sequence)

    @classmethod
    def frame_printer(self, frame):
        """
        pretty printer for a frame
        """

        for row in frame:
            row_output = ""
            for voxel in row:
                if voxel != VoxelState.NO_FILL.value:
                    row_output += "-"
                else:
                    row_output += " "

            print(row_output)


class Simulator:

    def __init__(self, width, height, arm, voxel_size = .01, start = np.array([1,2]), goal = np.array([2,1])):

        # these are in meters
        self.width = width
        self.height = height

        self.arm = arm
        self.start = start
        self.goal = goal

        # simulation init
        self.running = False
        self.fps = 100
        self.voxel_size = voxel_size # in meters

        self.check_conditions()
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

    def generate_voxels(self):
        """
        Generates a 2D grid of voxels based on the given width, height, and voxel size.
        """

        num_hor_vox = int(Decimal(str(self.width)) // Decimal(str(self.voxel_size)))
        num_vert_vox = int(Decimal(str(self.height)) // Decimal(str(self.voxel_size)))
        self.voxels = np.ones([num_hor_vox, num_vert_vox]) * VoxelState.NO_FILL.value

    def run(self, sim_time=10):
        """
        run the sim
        """

        self.running = True
        frame_time = 1/self.fps
        total_time = 0

        while self.running:

            self.arm.state_update(frame_time)
            self.fill_arm_voxels()
            self.recording.record_frame(self.voxels)

            # timing, can put in while loop if nothing else breaks the total time?
            total_time += frame_time
            if total_time >= sim_time: self.running = False

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
    
    @staticmethod
    def raycast(grid, start, end, voxel_size):
        """
        Perform a raycast on a grid of voxels and collect all collisions.
        Optimized version of the 'fast voxel traversal for ray tracing' algorithm.

        Parameters:
        - grid: 2D list of Voxel objects.
        - start: (x0, y0) start point of the ray.
        - end: (x1, y1) end point of the ray.
        - voxel_size: Size of each voxel.

        Returns:
        - List of Voxel objects that the ray intersects.
        """
        x0, y0 = start
        x1, y1 = end

        # Convert world coordinates to grid indices
        x0_idx = int(x0 // voxel_size)
        y0_idx = int(y0 // voxel_size)
        x1_idx = int(x1 // voxel_size)
        y1_idx = int(y1 // voxel_size)

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

        max_steps = int(math.sqrt((x1_idx - x0_idx)**2 + (y1_idx - y0_idx)**2)) + 1
        row_max = len(grid[0])
        col_max = len(grid)

        collided_voxels = set()  # Use a set for fast membership checking

        # DDA Loop Traversal
        for _ in range(max_steps):
            # Check if within bounds
            if 0 <= vox_row < row_max and 0 <= vox_col < col_max:
                vox_ids = (vox_col, vox_row)
                if vox_ids not in collided_voxels:
                    collided_voxels.add(vox_ids)
            else:
                break  # Stop if outside the grid

            # Move to the next voxel
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                vox_row += step_x
            else:
                t_max_y += t_delta_y
                vox_col += step_y

        return list(collided_voxels)


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

    @staticmethod
    def run_single_simulation(sim_id, sim_time, width=4.1, height=4.1, voxel_size=0.05, entry_point=None):
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
        arm = Arm(x0=width / 2, y0=height / 2, l1=1, l2=1, m1=1, m2=1, g=-9.8)
        sim = Simulator(width, height, arm, voxel_size)
        recording = sim.run(sim_time)
        recording.save(sim_id, entry_point)
        return sim_id, recording

    def batch_process(self, entry_point):
        """
        Run all simulations in parallel and collect the results.
        
        Returns:
        - A dictionary of {simulation_id: recording}.
        """
        results = {}
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Prepare simulation arguments
            tasks = [
                (sim_id, self.sim_time, 4.1, 4.1, 0.05, entry_point)  # Default params for each simulation
                for sim_id in range(self.num_sims)
            ]
            # Run simulations in parallel
            for sim_id, recording in pool.starmap(BatchProcessor.run_single_simulation, tasks):
                results[sim_id] = recording

        return results


if __name__ == "__main__":
    pass