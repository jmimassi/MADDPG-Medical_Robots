# MADDPG for Medical Robots

## Description

This project aims to develop a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm for controlling medical robots to find bodies and then heal them. The goal is to enable collaborative decision-making and coordination among multiple medical robots in order to improve efficiency and effectiveness in finding and healing.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Content](#content)

## Installation

To install and set up the project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/MADDPG-Medical_Robots.git`
2. Navigate to the project directory: `cd MADDPG-Medical_Robots`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

To use the project, follow these guidelines:

Our model is trained to handle a maximum of 2 agents and 2 bodies to find. These configurations are directly testable. However, if you need more configurations, you will need to train the model yourself.

1. To test the pretrained 1 agents and 1 bodies : `python3 test.py`
2. To test the other pretrained configuration you can change the number and bodies in the file `test.py` and change variables `n_agents = ... (max 2)` and `n_bodies = ... (max 2)`

3. To train other configuration, do the same step in the `train.py` file and run `python3 train.py`

## Content

- **Checkpoints** contains the value of the weights for each configurations of {agent-bodies}
- **Data** contains the rewards per agent for each 10 trajectories
- **environment** contains the continuous grid world environment
- **maddpg** contains the MADDPG algorithm  
