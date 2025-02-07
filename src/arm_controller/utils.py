from datetime import datetime
import os
import cProfile
import pstats
import re

ENTRY_POINT = None
DATA_FOLDER = "data/sim_data" # Holy god this should be a pathlike object
MODEL_FOLDER = "data/model_data" # this too

def pretty_date():
    """
    this godawful thing makes a date that can be saved as file name
    """
    return f"{datetime.now()}".replace(" ", "_").rsplit(".", 1)[0].replace(":", "_")

def set_entry_point(path_like):
    """
    there is definitely a better way to do this, but this sets the global for the entry point
    """
    global ENTRY_POINT
    ENTRY_POINT = path_like

def get_entry_point():
    """
    returns entry point global
    """
    global ENTRY_POINT
    return ENTRY_POINT

def get_data_folder():
    """
    honey where is my super suit? 
    """
    global ENTRY_POINT, DATA_FOLDER
    return ENTRY_POINT.joinpath(DATA_FOLDER)

def clear_old_data():
    """
    clear old data from data folder
    """

    for sim_data_path in get_data_folder().iterdir():
        if sim_data_path.is_file() and sim_data_path.suffix == ".npz":
            os.remove(sim_data_path)

def clear_old_model():
    """
    delete the old model
    """
    print("clear_old_model NOT IMPLEMENTED")

def get_model_folder():
    """
    get the folder that the model is supposed to live in
    """
    global ENTRY_POINT, MODEL_FOLDER
    return ENTRY_POINT.joinpath(MODEL_FOLDER)

def get_most_recent_model():
    """
    returns the path for the most recent model
    """

    base_name = "frame_prediction_model"
    pattern = rf"{base_name}_(\d{{1,3}})\.pth"

    most_recent_model = None
    highest_id = -1
    for file in get_model_folder().iterdir(): # TODO: yea this is not my best work
        if file.suffix == ".pth":
            match_file = re.search(pattern, str(file))
            
            if match_file:
                model_id = int(match_file.group(1)) 

                if model_id > highest_id:
                    most_recent_model = file
                    highest_id = model_id

    return most_recent_model

            




def profile_func(func, *args):
    """
    profiler
    """

    with cProfile.Profile() as pr:
        func(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
