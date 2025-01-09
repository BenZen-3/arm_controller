import numpy as np
import pygame
import time
from enum import Enum, auto
from . import Arm
import math
from datetime import datetime
from decimal import Decimal

import cProfile
import pstats

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (0 ,    0,   0)

class VoxelState(Enum):
    NO_FILL = 0
    FILLED_ARM = 1
    FILLED_OBSTACLE = 2


class Voxel:

    def __init__(self, x, y, voxel_size, state = VoxelState.NO_FILL):
        
        # the CENTER coordinates
        self.x = x
        self.y = y
        self.voxel_size = voxel_size # the width and height of the voxel
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if not isinstance(value, VoxelState):
            raise TypeError("Name must be a VoxelState")
        self._state = value

    def draw(self, screen):
        """
        draw the voxel
        """

        if self._state == VoxelState.NO_FILL:
            return # background is black already :)
        elif self._state == VoxelState.FILLED_ARM:
            self.draw_colored_vox(screen, GREEN)
        elif self._state == VoxelState.FILLED_OBSTACLE:
            self.draw_colored_vox(screen, RED)

    def draw_colored_vox(self, screen, color):
        """
        draws the voxel as a specific color
        """

        left = self.x - self.voxel_size/2
        top = self.y - self.voxel_size/2
        rect = pygame.Rect(left, top, self.voxel_size, self.voxel_size)
        pygame.draw.rect(screen, color, rect)

    # @classmethod
    # def draw_cell_value(self, value, screen, voxel_size):

    #     if value == VoxelState.NO_FILL.value:
    #         return # background is black already :)
    #     elif value == VoxelState.FILLED_ARM.value:
    #         self.draw_colored_cell(screen, GREEN)
    #     elif value == VoxelState.FILLED_OBSTACLE.value:
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

        # print('converting recording')
        # voxel_sequence = recording.convert_to_voxel_seq()
        # print(np.shape(voxel_sequence))
        # print('here')

        for frame in recording.frame_sequence:#voxel_sequence:
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
                if voxel.state != VoxelState.NO_FILL:
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

        # data = []

        # for voxel_row in voxel_grid:
        #     data_row = []
        #     for voxel in voxel_row:
        #         data_row.append(voxel.state.value)
        #     data.append(data_row)

        self.frame_sequence.append(np.copy(voxel_grid))

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
                    voxel = Voxel(x, y, self.voxel_size, VoxelState.FILLED_ARM)
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
        # self.pixel_per_meter = 100 # OLD REMOVE

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
        # print(VoxelState.NO_FILL.value)
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

            # timing
            total_time += frame_time
            if total_time >= sim_time: self.running = False

        return self.recording

    def arm_joint_cartesian(self): # OLD REMOVE
        """
        get the arm joint positions in cartesian
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

        filled = self.check_arm_occupancy()
        self.voxels *= VoxelState.NO_FILL.value # remove all previous filled areas

        # print(len(filled))

        for location in filled:
            # print(location)
            x,y = location
            self.voxels[x,y] = VoxelState.FILLED_ARM.value

    def check_arm_occupancy(self):
        """
        Checks where the arm is by casting a ray along each of the arm linkages
        """

        base, j1, j2 = self.arm.cartesian_joint_locations()
        # print(base, j1, j2)

        # print(np.shape(self.voxels))

        # v=self.raycast(self.voxels, [3,2], [4,4], self.voxel_size)
        # return v

        v1 = self.raycast(self.voxels, base, j1, self.voxel_size)
        v2 = self.raycast(self.voxels, j1, j2, self.voxel_size)
        return v2 + v1

    def vox_pos(self, id_x, id_y):
        """
        get the voxel's cartesian position based on its index
        """
        x = id_x * self.voxel_size
        y = id_y * self.voxel_size
        return (x, y)

    def raycast(self, grid, start, end, voxel_size):
        """
        Perform a raycast on a grid of voxels and collect all collisions.
        Algorithm: modified 'fast voxel traversal for ray tracing' (might contain mistakes)

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

        # print(f"{x0=}")
        # print(f"{y0=}")
        # print(f"{x0_idx=}")
        # print(f"{y0_idx=}")
        # print(f"{x1=}")
        # print(f"{y1=}")

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

        collided_voxels = []

        ray_len = 0.0
        max_ray_len = math.sqrt(dx**2 + dy**2)
        # print(f"{max_ray_len=}")

        # DDA Traversal Loop
        while ray_len < max_ray_len:

            # Check if within bounds
            if (0 <= vox_row < len(grid[0])) and (0 <= vox_col < len(grid)):
                voxel = self.vox_pos(vox_row, vox_col)#grid[current_y][current_x] # YEAH ITS A LITTLE BIT BACKWARDS I KNOW
                # print(voxel)

                if (vox_col, vox_row) not in collided_voxels:
                    collided_voxels.append((vox_col, vox_row))

            else:
                break # I can't think of how a ray would exit and then return to the grid?

            # yeah so like uhh if its longer than it should be then please kill this loop
            ray_len = math.sqrt((voxel[0]-x0)**2 + (voxel[1]-y0)**2)
            # print(f"here with ray len at {ray_len} and max {max_ray_len} and {voxel=}")

            if ray_len > max_ray_len:
                break

            # Move to the next voxel
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                vox_row += step_x
            else:
                t_max_y += t_delta_y
                vox_col += step_y

        # print(f"returned a collided voxel array or length: {len(collided_voxels)}")
        # print(self.vox_pos(loc[0], loc[1]) for loc in collided_voxels)
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

        for x in range(0, self.width, self.voxel_size):
            for y in range(0, self.height, self.voxel_size):
                rect = pygame.Rect(x, y, self.voxel_size, self.voxel_size)
                pygame.draw.rect(self.screen, WHITE, rect, 1)



if __name__ == "__main__":

    arm = Arm(x0=0, y0=0, l1=2, l2=2, m1=1, m2=1, g=-9.8)
    sim = Simulator(800, 600, "Arm Simulator", arm)


    with cProfile.Profile() as pr:
        sim.run()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()

    
     