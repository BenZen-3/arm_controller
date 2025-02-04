from .simulator import BatchProcessor, Recording
from .learning import  train, test, full_prediction
from .player import SimulationPlayer
import argparse
from . import utils
import time
from pathlib import Path
import cProfile
import pstats
import os
import numpy as np


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# DEV ONLY
def profile_func(func, *args):
    with cProfile.Profile() as pr:
        func(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()

def main():
    """
    main program start
    """

    # funny nonsense for globals
    utils.set_entry_point(Path.cwd().parent)

    parser = argparse.ArgumentParser(description="Arm controller Package")
    parser.add_argument(
        '--mode', 
        choices=['generator', 'trainer', 'predictor'], 
        default='generator',
        help='Choose which module to run. Generate data, train on data, or predict a sequence from a random starting state'
    )

    args = parser.parse_args()
    start = time.time()

    if args.mode == 'generator':
        generate_data()
    elif args.mode == 'trainer':
        train_model()
    elif args.mode == 'predictor':
        predict()

    print(f"Total time: {round(time.time() - start, 2)} seconds")

def generate_data(num_sims=200, sim_time=20):
    """
    generate data for training

    num_sims is the number of sim runs
    sim_time is the length of the simulation
    """

    print("Generating data")
    
    BatchSim = BatchProcessor(num_sims, sim_time)
    BatchSim.batch_process()

def train_model():
    print("Training model mode")
    entry_point = Path.cwd().parent # eventually clean up this weird path thing

    train(entry_point)

def test_model():
    print("testing model")

    model_save_path = "video_conv3d.pth"
    data_path = "test_data"
    test(model_save_path, data_path, entry_point = Path.cwd().parent)

    playback_file = "real.npy"
    recording = Recording()
    recording.init_from_file(playback_file)

    player = SimulationPlayer(800, 800)
    player.play(recording)

def predict():
    print("predicting future frames")

    _, recording = BatchProcessor.run_single_simulation(999,10,save=False)

    model_save_path = "video_conv3d.pth"
    initial_frames = recording.frame_sequence[:10]
    print(np.shape(initial_frames))
    frames = full_prediction(model_save_path, initial_frames, num_future_frames=100)
    # frames = np.append(initial_frames, frames)

    prediction = Recording()
    prediction.fps = 10
    prediction.frame_sequence = frames

    player = SimulationPlayer(800, 800)
    player.play(prediction)

    # _, recording = BatchProcessor.run_single_simulation(999,1,save=False)
    # player = SimulationPlayer(800, 800)
    # player.play(recording)


if __name__ == "__main__":
    main()






















# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# # DEV ONLY
# def profile_func(func, *args):
#     with cProfile.Profile() as pr:
#         func(*args)

#     stats = pstats.Stats(pr)
#     stats.sort_stats(pstats.SortKey.TIME)

#     stats.print_stats()

# def main():
#     parser = argparse.ArgumentParser(description="Arm controller Package")
#     parser.add_argument(
#         '--mode', 
#         choices=['simulator', 'generator', 'trainer', 'tester', 'predictor'], 
#         default='test_model',
#         help='Choose which module to run'
#     )
#     args = parser.parse_args()
    
#     if args.mode == 'simulator':
#         simulator_mode()
#     elif args.mode == 'generator':
#         data_generator_mode()
#     elif args.mode == 'trainer':
#         train_model()
#     elif args.mode == 'tester':
#         test_model()
#     elif args.mode == 'predictor':
#         predict()

# def simulator_mode():
#     print("Simulator mode")

#     _, recording = BatchProcessor.run_single_simulation(999,10,save=False)

#     # playback_file = "real.npy"
#     # recording = Recording()
#     # recording.init_from_file(playback_file)

#     player = SimulationPlayer(800, 800)
#     player.play(recording)

# def data_generator_mode():
#     print("Data Collection mode")
#     start = time.time()
#     entry_point = Path.cwd().parent # eventually clean up this weird path thing
    
#     BatchSim = BatchProcessor(200, 20)
#     BatchSim.batch_process(entry_point) 

#     print(f"Total batch sim time: {round(time.time() - start, 2)} seconds")

# def train_model():
#     print("Training model mode")
#     entry_point = Path.cwd().parent # eventually clean up this weird path thing

#     train(entry_point)

# def test_model():
#     print("testing model")

#     model_save_path = "video_conv3d.pth"
#     data_path = "test_data"
#     test(model_save_path, data_path, entry_point = Path.cwd().parent)

#     playback_file = "real.npy"
#     recording = Recording()
#     recording.init_from_file(playback_file)

#     player = SimulationPlayer(800, 800)
#     player.play(recording)

# def predict():
#     print("predicting future frames")

#     _, recording = BatchProcessor.run_single_simulation(999,10,save=False)

#     model_save_path = "video_conv3d.pth"
#     initial_frames = recording.frame_sequence[:10]
#     print(np.shape(initial_frames))
#     frames = full_prediction(model_save_path, initial_frames, num_future_frames=100)
#     # frames = np.append(initial_frames, frames)

#     prediction = Recording()
#     prediction.fps = 10
#     prediction.frame_sequence = frames

#     player = SimulationPlayer(800, 800)
#     player.play(prediction)

#     # _, recording = BatchProcessor.run_single_simulation(999,1,save=False)
#     # player = SimulationPlayer(800, 800)
#     # player.play(recording)


# if __name__ == "__main__":
#     main()