import argparse
from .simulator import BatchProcessor, Recording
from .learning import  train, test
from .player import SimulationPlayer
import time
from pathlib import Path
import cProfile
import pstats
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def profile_func(func, *args):
    with cProfile.Profile() as pr:
        func(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()

def main():
    parser = argparse.ArgumentParser(description="Arm controller Package")
    parser.add_argument(
        '--mode', 
        choices=['simulator', 'generator', 'trainer', 'tester'], 
        default='test_model',
        help='Choose which module to run'
    ) # simulator, generator, trainer, tester
    args = parser.parse_args()
    
    if args.mode == 'simulator':
        simulator_mode()
    elif args.mode == 'generator':
        data_generator_mode()
    elif args.mode == 'trainer':
        train_model()
    elif args.mode == 'tester':
        test_model()

def simulator_mode():
    print("Simulator mode")

    _, recording = BatchProcessor.run_single_simulation(999,10,save=False)

    # playback_file = "real.npy"
    # recording = Recording()
    # recording.init_from_file(playback_file)

    player = SimulationPlayer(800, 800)
    player.play(recording)

def data_generator_mode():
    print("Data Collection mode")
    start = time.time()
    entry_point = Path.cwd().parent # eventually clean up this weird path thing
    
    BatchSim = BatchProcessor(80, 20)
    BatchSim.batch_process(entry_point) 

    print(f"Total batch sim time: {round(time.time() - start, 2)} seconds")

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


if __name__ == "__main__":
    main()