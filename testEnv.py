#@title All `dm_control` imports required for this tutorial

# The basic mujoco wrapper.
from dm_control import mujoco, suite

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation


import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image



from dm_control import suite
import numpy as np
from collections import OrderedDict

def process_state(state):
    if isinstance(state, OrderedDict):
        if 'orientations' in state and 'height' in state and 'velocity' in state:
            orient = state['orientations']
            height = state['height']
            velocity = state['velocity']
            if np.isscalar(height):
                height = np.array([height])
            out = np.concatenate((orient, height, velocity))
            return out
    elif isinstance(state, np.ndarray) and state.shape == (24,):
        return state
    elif hasattr(state, 'observation') and isinstance(state.observation, OrderedDict):
        observation = state.observation
        if 'orientations' in observation and 'height' in observation and 'velocity' in observation:
            orient = observation['orientations']
            height = observation['height']
            velocity = observation['velocity']
            if np.isscalar(height):
                height = np.array([height])
            out = np.concatenate((orient, height, velocity))
            return out
    else:
        raise ValueError("Input state must be either an OrderedDict with keys 'orientations', 'height', and 'velocity', a numpy ndarray of shape (24,), or a TimeStep object with a valid observation.")




# Load one task:
env = suite.load(domain_name="cheetah", task_name="run")
#env = suite.walker.stand(time_limit=float('inf'))
# Iterate over a task set:
#for domain_name, task_name in suite.BENCHMARKING:
#  env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()

while True:
  random_state = np.random.RandomState()
  action = random_state.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)

  state = env.step(action)
  print(state)
  #obs = process_state(state)
  #reward = state.reward
  #doneFlag = state.last()

  #print(action,reward)
  break






# while not time_step.last():
#   action = np.random.uniform(action_spec.minimum,
#                              action_spec.maximum,
#                              size=action_spec.shape)
#   time_step = env.step(action)

#   state = process_state(time_step)

#   #print(type(state))
#   #print(state.shape)



#   break


# # #print("Observation keys:", time_step.observation.keys())
'''


# Cheetah Run

time_step.reward        0.012325912320161307    
time_step.discount      1.0 
time_step.observation   OrderedDict([
                        ('orientations', array([-0.41767168,  0.90859802,  0.5998677 ,  0.80009921, -0.37546694,
                                                    0.92683579,  0.16333507,  0.98657065,  0.43796533,  0.89899186,
                                                -0.19521768,  0.98075994, -0.15001222,  0.98868414])), 
                    ('height', 0.194405287018395), 
                    ('velocity', array([ 0.36757114, -0.22749672,  0.96173696,  2.08964333, -2.06501847,
                                            -5.18653552,  2.3555689 , -0.85863744,  1.02580639]))])

'''


'''
# Walker Stand
0.00927785532638026 
1.0 OrderedDict([
('orientations', array([-0.57450392,  0.81850183,  0.82107972,  0.57081353, -0.27540658,
                        0.96132784, -0.90537059,  0.4246223 ,  0.71074772,  0.703447  ,
                    -0.4323765 ,  0.90169316,  0.42935619,  0.90313524])), 
('height', 0.25500698840602065), 
('velocity', array([  0.09958946,  -1.44132875,  -1.31445721,   4.36834458,
                        -9.46918569,   0.83601598,   5.95773544, -14.99326023,
                        17.72638063]))])
'''
