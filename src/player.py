import pygame
import os
import time
from . import Recording, VoxelState


WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (0 ,    0,   0)


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

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

        for frame in recording.frame_sequence:#voxel_sequence:
            # print(type(frame))

            frame_start = time.time()

            self.check_quit()
            # self.draw_frame(frame)
            Recording.frame_printer(frame)

            time_taken = time.time() - frame_start
            pause_time = frame_time - time_taken
            if pause_time > 0:
                # time.sleep(frame_time - time_taken)
                pass
            else:
                print("WAAAAAAAAAAA")

            if not self.running: # probably go back to a while(self.running) later
                break

        pygame.quit()

    def draw_frame(self, frame): # TODO: This doesnt work at all

        # print(np.shape(frame))
        
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
