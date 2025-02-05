from datetime import datetime
import os
import cProfile
import pstats

ENTRY_POINT = None
DATA_FOLDER = "data" # Holy god this should be a pathlike object

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

def profile_func(func, *args):
    """
    profiler
    """

    with cProfile.Profile() as pr:
        func(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()
