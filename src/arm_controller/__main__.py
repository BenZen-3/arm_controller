import argparse
import time
from pathlib import Path
import os
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from .simulator import BatchProcessor, Recording
from .learning import  main_train, main_predict
from .player import SimulationPlayer
from . import utils

def main():
    """
    main program start
    """

    parser = argparse.ArgumentParser(description="Arm controller Package")
    parser.add_argument(
        '--mode', 
        choices=['generator', 'trainer', 'predictor', 'playback', "auto_idiot"], 
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
    elif args.mode == 'playback':
        playback_recording()
    elif args.mode == 'auto_idiot':
        auto_idiot()
        

    print(f"Total time: {round(time.time() - start, 2)} seconds")

def generate_data(num_sims=500, sim_time=20, clear_prev_data=True):
    """
    generate data for training

    num_sims is the number of sim runs
    sim_time is the length of the simulation
    """

    if clear_prev_data and input("Clearing Previous Data. \"N\" to stop. Enter to continue. ") != "N":
        utils.clear_old_data()
    
    print("Generating data")
    BatchSim = BatchProcessor(num_sims, sim_time)
    BatchSim.batch_process()

def train_model(use_stored_model=False):
    """
    train the model!
    
    params here
    """

    if not use_stored_model and input("Making fresh model. \"OLD\" to use existing model. Enter to continue. ") == "OLD":
        use_stored_model = True
        # logic can be cleaned. too lazy rn

    print("Training model")
    main_train(use_stored_model)

def predict():
    print("Predicting future frames")

    _, recording = BatchProcessor.run_single_simulation(0,1)
    print(np.shape(recording.frame_sequence))

    predicted_frames = main_predict(recording.get_float_frame_seq(), 100)
    # print(np.size(recording.get_float_frame_seq()))
    recording._frame_sequence = np.concatenate((np.copy(recording.frame_sequence), predicted_frames))
    # recording._frame_sequence = predicted_frames[0:1] #
    # recording._frame_sequence = predicted_frames # TODO: super jank, but saves the fps data soooo who cares for now

    recording.fps = 2

    player = SimulationPlayer(800, 800)
    # time.sleep(2)
    player.play(recording)

# this is temp and this is trash code
def playback_recording(rec_num=9):
    print("Playing recording")

    recs = []
    for sim_data_path in utils.get_data_folder().iterdir():
        if sim_data_path.is_file() and sim_data_path.suffix == ".npz":
            rec = Recording()
            rec.init_from_file(sim_data_path)
            recs.append(rec)

    play_me = recs[rec_num]

    print(type(play_me.frame_sequence[0]))
    import numpy as np
    print(np.max(play_me.frame_sequence[0]))

    player = SimulationPlayer(800, 800)
    player.play(play_me)

def auto_idiot():

    num_runs = 100
    
    num_sims = 200
    sim_time = 20

    for i in range(num_runs):

        utils.clear_old_data()
        BatchSim = BatchProcessor(num_sims, sim_time)
        BatchSim.batch_process()
        print(f"Run Number {i}")
        time.sleep(5)
        main_train(use_stored_model=True)


if __name__ == "__main__":

    # funny nonsense for globals
    utils.set_entry_point(Path.cwd().parent)

    main()
