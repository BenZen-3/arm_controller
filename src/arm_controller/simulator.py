import numpy as np
import pygame
import time
from enum import Enum, auto
from . import Arm
import math
from datetime import datetime

import cProfile
import pstats

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (0 ,    0,   0)

class CellState(Enum):
    NO_FILL = 0
    FILLED_ARM = 1
    FILLED_OBSTACLE = 2


class Voxel:

    def __init__(self, x, y, block_size, state = CellState.NO_FILL):
        
        # the CENTER coordinates
        self.x = x
        self.y = y
        self.block_size = block_size # the width and height of the cell
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if not isinstance(value, CellState):
            raise TypeError("Name must be a CellState")
        self._state = value

    def draw(self, screen):
        """
        draw the cell
        """

        if self._state == CellState.NO_FILL:
            return # background is black already :)
        elif self._state == CellState.FILLED_ARM:
            self.draw_colored_cell(screen, GREEN)
        elif self._state == CellState.FILLED_OBSTACLE:
            self.draw_colored_cell(screen, RED)

    def draw_colored_cell(self, screen, color):
        """
        draws the cell as a specific color
        """

        left = self.x - self.block_size/2
        top = self.y - self.block_size/2
        rect = pygame.Rect(left, top, self.block_size, self.block_size)
        pygame.draw.rect(screen, color, rect)

    # @classmethod
    # def draw_cell_value(self, value, screen, block_size):

    #     if value == CellState.NO_FILL.value:
    #         return # background is black already :)
    #     elif value == CellState.FILLED_ARM.value:
    #         self.draw_colored_cell(screen, GREEN)
    #     elif value == CellState.FILLED_OBSTACLE.value:
    #         self.draw_colored_cell(screen, RED)


class SimulationPlayer:
    
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.title = "Arm Simulator"
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

    def play(self, recording):
        """
        play a recording of a simulation
        """

        frame_time = 1/recording.fps
        self.running = True

        print('converting recording')
        voxel_sequence = recording.convert_to_voxel_seq()
        print(np.shape(voxel_sequence))
        print('here')

        for frame in voxel_sequence:
            # print(type(frame))

            frame_start = time.time()

            self.check_quit()
            # self.draw_frame(frame)
            Recording.frame_printer(frame)

            time_taken = time.time() - frame_start
            pause_time = frame_time - time_taken
            if pause_time > 0:
                time.sleep(frame_time - time_taken)
            else:
                print("WAAAAAAAAAAA")

            if not self.running: # probably go back to a while(self.running) later
                break

        pygame.quit()

    def draw_frame(self, frame):

        print(np.shape(frame))
        
        for voxel_row in frame:
            for voxel in voxel_row:
                if voxel.state != CellState.NO_FILL:
                    voxel.draw(self.screen)
                    # Voxel.draw_cell_value(voxel, self.screen)
                    # voxel.draw(self.screen)

        # pygame stuff
        pygame.display.flip()
        self.screen.fill(BLACK)

    def check_quit(self):
        """
        checks if pygame died or you press "Q"
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN: # Check if 'q' key was pressed
                if event.key == pygame.K_q: 
                    self.running = False


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

        # TODO: one-liner maybe?
        if name == None:
            self.name = f"simulation_data_{datetime.now()}"
        else:
            self.name = name

    def record_frame(self, voxel_grid):

        data = []

        for voxel_row in voxel_grid:
            data_row = []
            for voxel in voxel_row:
                data_row.append(voxel.state.value)
            data.append(data_row)

        self.frame_sequence.append(data)

    def save_as_file(self):
        pass

    def convert_to_voxel_seq(self):

        frame_vox_seq = []
        for frame in self.frame_sequence:

            voxel_sequence = []
            for row_num, data_row in enumerate(frame):
                voxel_row = []
                x = self.voxel_size /2 + row_num
                for col_num, value in enumerate(data_row):
                    y = self.voxel_size /2 + col_num
                    voxel = Voxel(x, y, self.voxel_size, CellState.FILLED_ARM)
                    voxel_row.append(voxel)
                voxel_sequence.append(voxel_row)
            frame_vox_seq.append(voxel_sequence)

        return frame_vox_seq

    @classmethod
    def frame_printer(self, frame):
        """
        pretty printer for a frame
        """

        for row in frame:
            row_output = ""
            for voxel in row:
                if voxel != CellState.NO_FILL.value:
                    row_output += "-"
                else:
                    row_output += " "

            print(row_output)

class Simulator:

    def __init__(self, width, height, arm, start = np.array([1,2]), goal = np.array([2,1])):

        # these are in pixels. TODO: get rid of any pixel values in this class 
        self.width = width
        self.height = height

        self.arm = arm
        self.start = start
        self.goal = goal

        # simulation init
        self.running = True
        self.fps = 100
        self.block_size = 10 # set the size of the grid in pix
        self.pixel_per_meter = 100
        self.generate_voxels()

        # recording
        self.recording = Recording()
        self.recording.init_for_recording(self.width, self.height, self.block_size, self.fps, self.arm)

    def generate_voxels(self):
        """
        Generates a 2D grid of voxels based on the given width, height, and block size.
        Each voxel is centered within its grid cell.
        """
        self.voxels = []  # Reset voxel grid

        # Iterate over rows (y-axis)
        for y in range(0, self.height, self.block_size):
            row = []  # Create a new row for each y
            # Iterate over columns (x-axis)
            for x in range(0, self.width, self.block_size):
                # Place voxel at the center of its block
                voxel = Voxel(
                    x + self.block_size / 2,
                    y + self.block_size / 2,
                    self.block_size
                )
                row.append(voxel)
            self.voxels.append(row)  # Add row to the grid

        self.voxels = np.asanyarray(self.voxels)

    def run(self, sim_time=10):
        """
        run the sim
        """

        frame_time = 1/self.fps
        total_time = 0

        while self.running:

            self.arm.state_update(frame_time)
            self.fill_arm_voxels()
            self.recording.record_frame(self.voxels)

            # timing
            total_time += frame_time
            if total_time >= sim_time: self.running = False

        return self.recording

    def arm_joint_pixels(self):
        """
        get the arm joint positions in pixels
        """

        offset = np.array([self.width/2, self.height/2])

        base, j1, j2 = self.arm.cartesian_joint_locations()
        arm_pix_width = round(self.arm.linkage_width*self.pixel_per_meter)

        base_pix = np.array(base*self.pixel_per_meter + offset).astype(int)
        j1_pix   = np.array(j1*self.pixel_per_meter + offset).astype(int)
        j2_pix   = np.array(j2*self.pixel_per_meter + offset).astype(int)

        return base_pix, j1_pix, j2_pix, arm_pix_width

    def fill_arm_voxels(self):
        """
        fill all my voxels
        """

        filled = self.check_grid_occupancy()

        for voxel in self.voxels.flatten():
            if voxel in filled: 
                voxel.state = CellState.FILLED_ARM
            else:
                voxel.state = CellState.NO_FILL # NEED TO UPDATE LATER WHEN OBSTACLES ARE INTRODUCED

    def check_grid_occupancy(self):
        """
        Checks where the arm is by casting a ray along each of the arm linkages
        """

        base_pix, j1_pix, j2_pix, arm_pix_width = self.arm_joint_pixels()

        v1 = self.raycast(self.voxels, base_pix, j1_pix, self.block_size)
        v2 = self.raycast(self.voxels, j1_pix, j2_pix, self.block_size)
        return v1 + v2

    def raycast(self, grid, start, end, block_size):
        """
        Perform a raycast on a grid of voxels and collect all collisions.
        Algorithm: modified 'fast voxel traversal for ray tracing' (might contain mistakes)

        Parameters:
        - grid: 2D list of Voxel objects.
        - start: (x0, y0) start point of the ray.
        - end: (x1, y1) end point of the ray.
        - block_size: Size of each voxel.

        Returns:
        - List of Voxel objects that the ray intersects.
        """
        x0, y0 = start
        x1, y1 = end

        # Convert world coordinates to grid indices
        x0_idx = int(x0 // block_size)
        y0_idx = int(y0 // block_size)

        dx = x1 - x0
        dy = y1 - y0

        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        t_max_x = ((x0_idx + (step_x > 0)) * block_size - x0) / dx if dx != 0 else float('inf')
        t_max_y = ((y0_idx + (step_y > 0)) * block_size - y0) / dy if dy != 0 else float('inf')

        t_delta_x = block_size / abs(dx) if dx != 0 else float('inf')
        t_delta_y = block_size / abs(dy) if dy != 0 else float('inf')

        current_x = x0_idx
        current_y = y0_idx

        collided_voxels = []

        ray_len = 0.0
        max_ray_len = math.sqrt(dx**2 + dy**2)

        # DDA Traversal Loop
        while ray_len < max_ray_len:

            # Check if within bounds
            if (0 <= current_x < len(grid[0])) and (0 <= current_y < len(grid)):
                voxel = grid[current_y][current_x] # YEAH ITS A LITTLE BIT BACKWARDS I KNOW

                if voxel not in collided_voxels:
                    collided_voxels.append(voxel)

            else:
                break # I can't think of how a ray would exit and then return to the grid?

            # yeah so like uhh if its longer than it should be then please kill this loop
            ray_len = math.sqrt((voxel.x-x0)**2 + (voxel.y-y0)**2)
            if ray_len > max_ray_len:
                break

            # Move to the next voxel
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                current_x += step_x
            else:
                t_max_y += t_delta_y
                current_y += step_y

        return collided_voxels


class Deprecated:

    def draw_arm(self): # its a draw so no
        """
        draws the arm
        """

        # get arm config in pixels
        base_pix, j1_pix, j2_pix, arm_pix_width = self.arm_joint_pixels()

        # arm links
        radius = round(arm_pix_width/2)
        pygame.draw.circle(self.screen, RED, base_pix, radius)
        pygame.draw.line(self.screen, RED, base_pix, j1_pix, arm_pix_width)
        pygame.draw.circle(self.screen, BLUE, j1_pix, radius)
        pygame.draw.line(self.screen, BLUE, j1_pix, j2_pix, arm_pix_width)
        pygame.draw.circle(self.screen, BLUE, j2_pix, radius)

    def draw_grid(self): # lord no
        """
        draws a grid
        """

        for x in range(0, self.width, self.block_size):
            for y in range(0, self.height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, WHITE, rect, 1)



if __name__ == "__main__":

    arm = Arm(x0=0, y0=0, l1=2, l2=2, m1=1, m2=1, g=-9.8)
    sim = Simulator(800, 600, "Arm Simulator", arm)


    with cProfile.Profile() as pr:
        sim.run()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()

    
     