import argparse
import time
from pathlib import Path
import os
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from .simulator import BatchProcessor, Recording
from .learning import  train, test, full_prediction
from .player import SimulationPlayer
from . import utils

def main():
    """
    main program start
    """

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

def generate_data(num_sims=200, sim_time=20, clear_prev_data=True):
    """
    generate data for training

    num_sims is the number of sim runs
    sim_time is the length of the simulation
    """

    if clear_prev_data and input("Clear Previous Data? \"N\" to stop. Enter to continue. ") != "N":
        utils.clear_old_data()
    
    print("Generating data")
    BatchSim = BatchProcessor(num_sims, sim_time)
    BatchSim.batch_process()

def train_model(clear_old_model=True):
    """
    train the model!
    
    params here
    """
    print("Training model mode")

    if clear_old_model and input("Clear Previous Model? \"Y\" to clear. Enter to continue. ") == "Y":
        utils.clear_old_model()
    

    # recs = []
    # for sim_data_path in utils.get_data_folder().iterdir():
    #     if sim_data_path.is_file() and sim_data_path.suffix == ".npz":
    #         rec = Recording()
    #         rec.init_from_file(sim_data_path)
    #         recs.append(rec)

    

    # entry_point = Path.cwd().parent # eventually clean up this weird path thing

    # train(entry_point)

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

    # funny nonsense for globals
    utils.set_entry_point(Path.cwd().parent)

    main()
