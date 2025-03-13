import argparse
import time
from pathlib import Path
import os
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from .simulator import BatchProcessor, Recording, run_controller_sim, generate_sim_data
from .language_generator import gather_llm_responses, load_all_llm_data
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
        choices=['generator', 'trainer', 'inference', 'playback', "auto_idiot", "controller"], 
        default='generator',
        help='Choose which module to run. Generate data, train on data, or predict a sequence from a random starting state'
    )

    args = parser.parse_args()
    start = time.time()

    if args.mode == 'generator':
        generate_data()
    elif args.mode == 'trainer':
        train_model()
    elif args.mode == 'inference':
        predict()
    elif args.mode == 'playback':
        playback_recording()
    elif args.mode == 'auto_idiot':
        auto_idiot()
    elif args.mode == 'controller':
        # llm_controller()
        gen_sim_data()

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

def predict(file=None):
    print("Predicting future frames")

    if file is None:
        _, recording = BatchProcessor.run_single_simulation(0,1.6)
    else:

        for data_file in utils.get_data_folder().iterdir():
            if data_file.suffix == ".pkl" and str(file) in str(data_file):
                recording = Recording()
                recording.init_from_file(data_file)
                break

    predicted_frames = main_predict(recording.get_float_frame_seq(), 1000)

    # TODO: super jank, but saves the fps data soooo who cares for now

    recording._frame_sequence = np.concatenate((np.copy(recording.frame_sequence), predicted_frames)) # append the real and predicted
    # recording._frame_sequence = predicted_frames[0:1] # just the first frame
    # recording._frame_sequence = predicted_frames # all the predicted

    # recording.fps = 2
    # recording.fps = .001

    player = SimulationPlayer(800, 800)
    time.sleep(10)
    player.play(recording)




# this is temp and this is trash code
def playback_recording(rec_num=485):
    print("Playing recording")

    recs = []
    for sim_data_path in utils.get_data_folder().iterdir():
        if sim_data_path.is_file() and sim_data_path.suffix == ".pkl":
            rec = Recording()
            rec.init_from_file(sim_data_path)
            recs.append(rec)
            
            if rec.sim_prompt:
                print(rec.sim_prompt)

    play_me = recs[rec_num]
    player = SimulationPlayer(800, 800)
    player.play(play_me)

def auto_idiot():

    num_runs = 100
    
    num_sims = 500
    sim_time = 20

    for i in range(num_runs):

        utils.clear_old_data()
        BatchSim = BatchProcessor(num_sims, sim_time)
        BatchSim.batch_process()
        print(f"Run Number {i}")
        time.sleep(5)
        main_train(use_stored_model=True)

def llm_controller():

    for i in range(10):
        gather_llm_responses()

    data = load_all_llm_data()
    run_controller_sim(data)

def gen_sim_data():

    utils.clear_old_data()
    data = load_all_llm_data()
    generate_sim_data(data)




if __name__ == "__main__":

    # funny nonsense for globals
    utils.set_entry_point(Path.cwd().parent)

    main()


"""
Active Notes:

For MSE backed recursive loss function: Epoch [10000/10000] Progress: 0%, Loss: 0.0024128351360559464
    first frame is dead on. after that the arm just slowly dissapears until there is no arm left :(
    16 and 32 model

Same as before, except 32 and 64 model: Epoch [7146/10000] Progress: 0%, Loss: 0.002413766225799918
    Loss is about the same

with 10 label frames: Epoch [6067/10000] Progress: 0%, Loss: 0.004137696232646704
    Looks better than before for a little bit longer now

    

I had a random idea, but can't seem to find anything on it (probably because I do not know what to google). 
Has anyone tried using diffusion models with only partially adding noise to an image? 
I'm thinking in terms of combining a diffusion policy with a physics simulation.

You'd be able to 'ground' the output of the network - because the physics simulation has pretty okay results in the short term - you can ground your output 
by not adding noise to the known ouput of the network - or even leave some noise based on how confident the physics result is

Then you can leave the important parts of the image fully noisy - this would allow for the model to generate from scratch
in the areas where there is a lot of noise, while leaving behind the areas that the physics engine took care of

maybe this is faster and more grounded - then your planning and manipulation and everything that your diffusion policy does can be 
physically grounded by a sim - a great benefit of an MPC - while having the excellent generative abilities and creative hallucinations of a large model


What I have tried so far: 
    Small, Medium, and Larger auto-encoder CNN

What others have done in the past for similar problems:
    Conv-LSTMs

What I want to do in the future:
    diffusion transformer!

    

I can generate json for an arm's path
I can run json for a single path

I need to be able to batch process these and make a bunch ton of paths from a bunch of prompts

Dont give it state at train - give it state only during use!

Can the diffusion process be guided by what the robot should look like? That way it actually outputs the correct robot...


Current status:
    so the LLM is actually sooo bad at making the robot do things that make sense.
    I need to give it my own shapes with parameterized functions OR G-code possibly

New Idea:
    3D gaussian diffusion.
    even 4D gaussian diffusion?

Can I structure diffusion to layer on different shapes?
    What if it starts with big shapes and then they get smaller?
    Generates the big structure first and the details later!

End Goal:
    Diffusion model for 3D or 4D Gaussians
    Hierarchical diffusion
        start with big gaussians
        move to smaller gaussians
    selective gaussian placement - where do they start?
    guided diffusion - how well does my diffusion line up with my robot? Also, you can guide it with physics!

    
Questions:
    What is a latent diffusion model and how is it any different?
    Should I use a variational autoencoder?

    
Where am I going
    I can make a GMM based off a few rectangles and gaussians if I want
    Now I should take the arm sim and conver that into moving rectangles
    Then convert the moving rectangles into moving GMM distributions
    train on the moving distributions auto regressively and see what happens

    after auto regression works
    try it as a diffusion policy!



    
What did I install:
    torch
    pygame
    dotenv
    google-genai
    matplotlib
    scipy
    scikit-learn

"""