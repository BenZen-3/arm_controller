# ArmController



Here's what's up right now. This codebase is a mess. Here's a breakdown of what needs to happen

Old Structure:

└── ./
    └── src
        └── arm_controller
            ├── __init__.py
            ├── __main__.py
            ├── arm.py
            ├── controller.py
            ├── gmm_estimation.py
            ├── language_generator.py
            ├── learning.py
            ├── player.py
            ├── simulator.py
            ├── testing_2.py
            ├── testing.py
            └── utils.py
    └── data
    └── tests

Installations so far (old):
    torch
    pygame
    dotenv
    google-genai
    matplotlib
    scipy
    scikit-learn


New Structure:

├── src/
    ├── arm_controller/
        ├── __init__.py             # its a package
        ├── __main__.py             # entry point. Run as module
        ├── utils.py                # generic utils
        |
        ├── core/
        │   └── publisher.py        # publisher base class  
        │   └── subscriber.py       # subscriber base class
        │   └── message_bus.py      # connect publishers and subscribers
        │   └── message_types.py    # common message types
        |
        ├── simulation/
        │   └── sim_manager.py      # runs the simulation(s) and publishes data to topics as needed
        │   └── arm.py              # describe arm dynamics and kinematics
        │   └── controller.py       # arm controllers (like joint space PD controller for example)
        │
        ├── data_synthesis/
        │   └── gmm_estimator.py    # estimates a 2 dof arm as a GMM
        │   └── data_diffuser.py    # converts stream of data into a diffusion-ready dataset
        |
        ├── learning/                
        │   ├── model.py            # contains the description of the model
        │   ├── train.py            # train the model
        │
        ├── visualization/
        │   └── gmm_plotting.py     # plotting for GMMs
        │   └── arm_plotting.py     # plotting of just the arm
|
├── data/                   # contains various kinds of data
│   ├── sim_data/
│   ├── model_data/
|
├── README.md
└── requirements.txt


Generic pipeline:

1) Description of arm, arm controller and scene
2) Simulation of arm in environment
    a) Simulator publishes scene data (arm joint locations, goals, etc)
    b) Controller subscribes to published data and publishes control signals
    c) Simulation takes published control signals and runs the arm
    d) Separate observers can subscribe to their needed topics (GMM estimator subs to joint states, state history collector subs to joint states)
3) once a simulation is run, the data can be handled by the data synthesis tools
    a) these tools are responsible for saving data for the learning pipeline
    b) the pattern for this is generator -> data_object (subclass of torch Dataset), data_object describes the simulation
4) using the data from the data synthesis tools, the learning pipeline can learn whatever it wants
    a) the data objects are unpickled and loaded as Dataset-like objects
    b) learning pipline will take these and learn the dataset
    c) the model file holds all things relevant to model architecture