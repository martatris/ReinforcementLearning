FrozenLake: Double DQN with Prioritized Experience Replay
=========================================================

This project demonstrates Double Deep Q-Learning (Double DQN) combined with
Prioritized Experience Replay (PER) on the FrozenLake-v1 environment from Gymnasium.

Features
--------
- Double DQN to reduce Q-value overestimation.
- Prioritized Experience Replay (PER) to sample informative transitions more often.
- Soft Target Updates for smoother training.
- Text-based rendering using the 'ansi' mode.
- GPU/CPU support with automatic device selection.

Requirements
------------
Python 3.8+ and the following packages:

    - gymnasium
    - torch
    - numpy

Files
-----
dqn_frozenlake_double_per.py - Main training script.

README.txt - This documentation file.

How to Run
----------
1. Install dependencies:

       pip install gymnasium torch numpy
   
3. Run the training script:

       python dqn_frozenlake_double_per.py

Algorithm Overview
------------------
Double DQN:

    - Uses separate networks for action selection and evaluation.
    
    - Reduces overestimation of Q-values and stabilizes training.

Prioritized Experience Replay:

    - Samples transitions with higher TD-error more frequently.
    
    - Applies importance sampling weights to correct bias.

Key Hyperparameters
-------------------

alpha: 0.6  (priority strength for PER)

beta:  0.4  (importance sampling correction)

gamma: 0.99 (discount factor)

tau:   0.01 (target network update rate)

batch_size: 64

learning_rate: 1e-3

epsilon_decay: 0.001

Expected Output Example
-----------------------

Training Double DQN + PER ...

Episode 200/2000, ε=0.818, avg_reward=0.26

Episode 400/2000, ε=0.670, avg_reward=0.48

...

Training complete!


Win rate (Double DQN + PER): 96.50%

Reached the goal!

Environment Description
-----------------------

FrozenLake-v1 is a simple 4x4 grid world:

    S = Start
    
    F = Frozen (safe)
    
    H = Hole (danger)
    
    G = Goal

The agent learns an optimal path from S → G without falling into holes.

Future Improvements
-------------------

- Extend to CartPole-v1 or LunarLander-v2

- Implement Dueling DQN architecture

- Add reward normalization and LR scheduling

- Visualize training metrics with matplotlib

License
-------
Author: Triston Aloyssius Marta
