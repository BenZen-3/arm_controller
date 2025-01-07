import numpy as np
import pygame
import time
from enum import Enum, auto
import math

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


class Voxel():

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

        if self._state == CellState.NO_FILL:
            return # background is black already :)
        elif self._state == CellState.FILLED_ARM:
            self.draw_colored_cell(screen, GREEN)
        elif self._state == CellState.FILLED_OBSTACLE:
            self.draw_colored_cell(screen, RED)

    def draw_colored_cell(self, screen, color):

        left = self.x - self.block_size/2
        top = self.y - self.block_size/2
        rect = pygame.Rect(left, top, self.block_size, self.block_size)
        pygame.draw.rect(screen, color, rect)


class Simulator(pygame.sprite.Sprite):


    def __init__(self, width, height, title, arm, start = np.array([1,2]), goal = np.array([2,1])):
        super().__init__()
        self.width = width
        self.height = height
        self.title = title
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

        self.arm = arm
        self.start = start
        self.goal = goal

        self.running = True
        self.fps = 100
        self.block_size = 10 # set the size of the grid in pix
        self.pixel_per_meter = 100
        self.generate_voxels()


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

    def run(self):

        frame_time = 1/self.fps
        
        start_time = time.time()

        while self.running:

            frame_start = time.time()
            self.check_quit()

            time_taken = time.time() - frame_start
            arm.state_update(frame_time)
            self.draw_all()
            time.sleep(frame_time - time_taken)

            # if time.time() - start_time > 10:
            #     self.running = False

        pygame.quit()

    def check_quit(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def draw_all(self):

        # self.draw_arm()
        self.draw_voxels()
        # self.draw_grid() # EXTREMELY inefficient

        # pygame stuff
        pygame.display.flip()
        self.screen.fill(BLACK)

    def arm_joint_pixels(self):

        offset = np.array([self.width/2, self.height/2])

        base, j1, j2 = arm.cartesian_joint_locations()
        arm_pix_width = round(arm.linkage_width*self.pixel_per_meter)

        base_pix = np.array(base*self.pixel_per_meter + offset).astype(int)
        j1_pix   = np.array(j1*self.pixel_per_meter + offset).astype(int)
        j2_pix   = np.array(j2*self.pixel_per_meter + offset).astype(int)

        return base_pix, j1_pix, j2_pix, arm_pix_width

    def draw_arm(self):

        # get arm config in pixels
        base_pix, j1_pix, j2_pix, arm_pix_width = self.arm_joint_pixels()

        # arm links
        radius = round(arm_pix_width/2)
        pygame.draw.circle(self.screen, RED, base_pix, radius)
        pygame.draw.line(self.screen, RED, base_pix, j1_pix, arm_pix_width)
        pygame.draw.circle(self.screen, BLUE, j1_pix, radius)
        pygame.draw.line(self.screen, BLUE, j1_pix, j2_pix, arm_pix_width)
        pygame.draw.circle(self.screen, BLUE, j2_pix, radius)

    def draw_grid(self):

        for x in range(0, self.width, self.block_size):
            for y in range(0, self.height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw_voxels(self):

        filled = self.check_grid_occupancy()

        for voxel in filled:
            voxel.state = CellState.FILLED_ARM
            voxel.draw(self.screen)

    def check_grid_occupancy(self):

        base_pix, j1_pix, j2_pix, arm_pix_width = self.arm_joint_pixels()

        v1 = raycast(self.voxels, base_pix, j1_pix, self.block_size)
        v2 = raycast(self.voxels, j1_pix, j2_pix, self.block_size)
        return v1 + v2


def raycast(grid, start, end, block_size):
    """
    Perform a raycast on a grid of voxels and collect all collisions.

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
    x1_idx = int(x1 // block_size)
    y1_idx = int(y1 // block_size)

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

##############################################################

class Arm():

    class ArmState:

        def __init__(self, x0=0.0, y0=0.0, theta1=0.0, theta2=0.0, theta1_dot=0.0, theta2_dot=0.0):
            self.x0 = x0
            self.y0 = y0
            self.theta1 = theta1
            self.theta2 = theta2
            self.theta1_dot = theta1_dot
            self.theta2_dot = theta2_dot

    def __init__(self, x0=0, y0=0, l1=1, l2=1, m1=1, m2=1, g=-9.8):
        
        # params
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.linkage_width = .32

        # carry the state in a separate instance for clarity between params and state vars
        self.state = self.ArmState(x0,y0)

    def dynamics(self):

        # state
        theta1 = self.state.theta1
        theta2 = self.state.theta2
        theta1_dot = self.state.theta1_dot
        theta2_dot = self.state.theta2_dot

        # arm params
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1 
        m2 = self.m2
        g = self.g

        # Mass Matrix, M(q)
        M = np.array([
            [m1 * l1**2 + m2 * (l1**2 + 2 * l1 * l2 * np.cos(theta2) + l2**2),
            m2 * (l1 * l2 * np.cos(theta2) + l2**2)],
            [m2 * (l1 * l2 * np.cos(theta2) + l2**2),
            m2 * l2**2]
        ])

        # Coriolis and Centripetal Matrix, C(q, q_dot)
        C = np.array([
            [-m2 * l1 * l2 * np.sin(theta2) * (2 * theta1_dot * theta2_dot + theta2_dot**2)],
            [m2 * l1 * l2 * theta1_dot**2 * np.sin(theta2)]
        ])

        # Gravity Vector, G(q)
        G = np.array([
            [(m1 + m2) * l1 * g * np.cos(theta1) + m2 * g * l2 * np.cos(theta1 + theta2)],
            [m2 * g * l2 * np.cos(theta1 + theta2)]
        ])

        return M, C, G

    def state_update(self, dt, U=np.array([[0], [0]])):
        def dynamics_wrapper(theta1, theta2, theta1_dot, theta2_dot):
            self.state.theta1 = theta1
            self.state.theta2 = theta2
            self.state.theta1_dot = theta1_dot
            self.state.theta2_dot = theta2_dot
            
            M, C, G = self.dynamics()
            M_inv = np.linalg.inv(M) # can i switch to a solve instead?
            q_dd = np.dot(M_inv, -C - G + U)
            
            return np.array([theta1_dot, theta2_dot, q_dd[0][0], q_dd[1][0]])
        
        # RK4 Integration
        state = np.array([self.state.theta1, self.state.theta2, self.state.theta1_dot, self.state.theta2_dot])
        
        k1 = dt * dynamics_wrapper(*state)
        k2 = dt * dynamics_wrapper(*(state + 0.5 * k1))
        k3 = dt * dynamics_wrapper(*(state + 0.5 * k2))
        k4 = dt * dynamics_wrapper(*(state + k3))

        state += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6
        
        self.state.theta1, self.state.theta2, self.state.theta1_dot, self.state.theta2_dot = state

    def cartesian_joint_locations(self):

        x1 = self.l1 * np.cos(self.state.theta1)
        y1 = self.l1 * np.sin(self.state.theta1)
        x2 = x1 + self.l2 * np.cos(self.state.theta1 + self.state.theta2)
        y2 = y1 + self.l2 * np.sin(self.state.theta1 + self.state.theta2)

        return np.array([self.state.x0,self.state.y0]), np.array([x1,y1]), np.array([x2,y2])


if __name__ == "__main__":

    arm = Arm(x0=0, y0=0, l1=2, l2=2, m1=1, m2=1, g=-9.8)
    sim = Simulator(800, 600, "Arm Simulator", arm)


    with cProfile.Profile() as pr:
        sim.run()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()

    
     