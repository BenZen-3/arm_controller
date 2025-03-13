import pygame
import time
import numpy as np

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (0 ,    0,   0)

class SimulationPlayer:
    
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.title = "Arm Simulator"
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

        self.history = [] # TODO: this really doesnt belong here. get rid of this

    def realtime_play(self, sim, trace=False): # TODO: This is a lot of dupe code
        """
        yeah it does that
        """

        # TODO: this is hardcoded
        num_vox_width, num_vox_height =  64, 64
        self.voxel_size = min(self.width // num_vox_width, self.height // num_vox_height)

        fps = 30
        frame_time = 1/fps
        self.running = True
        sim.running = True

        while self.running:
            frame_start = time.time()

            # all the sim stuff for running
            sim.control_arm(frame_time)
            sim.fill_arm_voxels()
            sim.recording.record_frame(sim.voxels)
            frame = sim.recording.frame_sequence[-1]/255 # get the final frame

            self.check_quit()
            #self.draw_frame(frame)

            if trace: 
                ee_pos = sim.arm.cartesian_EE_location()
                self.history.append(ee_pos)
                for point in self.history:
                    self.trace_arm_position(point)

            time_taken = time.time() - frame_start
            pause_time = frame_time - time_taken
            if pause_time > 0:
                pass
                #time.sleep(frame_time - time_taken)

            time.sleep(frame_time)

            if not self.running or not sim.running:
                break

        pygame.quit()

    def trace_arm_position(self, ee_pos):
        """
        yeah
        """

        ee_zeroed = np.array([ee_pos[0] - 2.1, ee_pos[1] - 2.1])
        pix_location =  ee_zeroed * 64 * 12 / 4.2 # num voxels, voxel size, size of screen in meters
        pygame.draw.circle(self.screen, RED, (pix_location[0] + self.width /2, self.height/2 - pix_location[1]), 10)

    def play(self, recording):
        """
        play a recording of a simulation
        """

        self.setup_to_play(recording)

        frame_time = 1/recording.fps
        self.running = True

        for frame in recording.frame_sequence:

            frame_start = time.time()

            self.check_quit()
            self.draw_frame(frame)

            time_taken = time.time() - frame_start
            pause_time = frame_time - time_taken
            if pause_time > 0:
                time.sleep(frame_time - time_taken)
                # pass

            if not self.running: # probably go back to a while(self.running) later
                break

        pygame.quit()

    def setup_to_play(self, recording):
        """
        setup
        """
        size = np.shape(recording.frame_sequence)
        num_vox_width = size[1]
        num_vox_height = size[2]
        self.voxel_size = min(self.width // num_vox_width, self.height // num_vox_height)

    def draw_frame(self, frame):

        # flip the frame so that it can be drawn from the lop left
        frame = np.flipud(frame)

        for vox_id_vert, voxel_row in enumerate(frame):
            for vox_id_hor, voxel in enumerate(voxel_row):
                # if round(voxel) != VoxelState.NO_FILL.value:
                if voxel > .2:

                    left = vox_id_hor * self.voxel_size
                    top = (vox_id_vert + 1) * self.voxel_size
                    rect = pygame.Rect(left, top, self.voxel_size, self.voxel_size)
                    color = self.voxel_color(WHITE, voxel)
                    pygame.draw.rect(self.screen, color, rect)             

        # pygame stuff
        pygame.display.flip()
        self.screen.fill(BLACK)

    def voxel_color(self, original_color, voxel_val):
        r, g, b = iter(original_color)                      # TODO: FIX THIS: OverflowError: cannot convert float infinity to integer HERE: r = max(0, min(255, round(r * voxel_val)))
        r = max(0, min(255, round(r * voxel_val))) # this is UGLY
        g = max(0, min(255, round(g * voxel_val)))
        b = max(0, min(255, round(b * voxel_val)))
        return (r,g,b)


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
