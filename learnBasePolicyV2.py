from DDPG import DDPG
from utils import ReplayBuffer
import torch
import numpy as np
import pickle
import argparse 
import time 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from dm_control import suite


parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action='store_true', help="Set train mode or evaluate")
args = parser.parse_args()
batch_size = 256

def clog_prob(val, mu=0, std=1):
    return np.sum(-np.log(np.sqrt(2*np.pi))-np.log(std**2)-(1/2*std)*np.square(val-mu))

def process_state(timeStep):
    try:
        #time_step = env.reset()
        state_observation = timeStep.observation
        orient = state_observation['orientations']
        height = state_observation['height']
        velocity = state_observation['velocity']
        if np.isscalar(height):
            height = np.array([height])

        state = np.concatenate((orient, height, velocity))
        return state
    except:
        return timeStep

# def fillBuffer(env, memory, action_space, length=1000, returnlp=True):
    
#     state = process_state(env.reset())
#     for _ in range(length):
#         action = np.random.normal(size=action_space)
#         if returnlp:
#             log_prob = clog_prob(action)
#         new_state, reward, done, _ = env.step(action)
#         new_state = process_state(new_state)
#         if returnlp:
#             memory.add((state, action, reward, new_state, log_prob, done))
#         else:
#             memory.add((state, action, reward, new_state, done))
#         state = new_state
#         if done:
#             state = process_state(env.reset())

# def recordEps(env, model, recordEps=True, save_path=None):

#     #  Records episode run by default, otherwise acts as a test run for the model
#     if recordEps:
#         rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, save_path)
#     state = env.reset()
#     r = 0
#     step = 0
#     while(True and step <= 5000):
#         tup = model.predict(np.expand_dims(state, axis=0), test=True)
#         if (type(tup) is tuple):
#             action, _ = tup
#         else:
#             action = tup
#         if recordEps:
#             rec.capture_frame()
#         state, reward, done, _ = env.step(action)
#         r += reward
#         step += 1
#         if done:
#             if recordEps:
#                 rec.close()
#             print("Reward at termination: {}".format(r))
#             print("Avg Reward: {}".format(r/step))
#             return

# def baseEp(env):

#     state = env.reset()
#     r = 0
#     while(True):
#         action = np.random.normal(size = env.action_space.shape[0])
#         state, reward, done, _ = env.step(action)
#         r += reward
#         if done:
#             print("Random: Reward at Termination: {}".format(r))
#             return

def runEp(env, memory, model, returnlp=False, returnReward=False):
    step = 0
    epochReward = []
    stepList = []
    epReward = 0
    
    state = process_state(env.reset())
    for _ in range(5000): # Note that 5000 was an arbitrary choice
        if not returnlp:
            action = model.predict(np.expand_dims(state, axis=0))
        else:
            #print(model.predict(np.expand_dims(state, axis=0)))
            action, log_prob = model.predict(np.expand_dims(state, axis=0))
        
        time_step = env.step(action)
        new_state = process_state(time_step)
        reward = time_step.reward

        # Check this
        done = False

        new_state = process_state(new_state)
        
        #  Add transition tuple to replay buffer
        if not returnlp:
            memory.add((state, action, reward, new_state, done))
        else:
            memory.add((state, action, reward, new_state, log_prob, done))
        state = new_state
        step += 1
        
        #  Train every 50 steps
        if (step%50) == 0:
            for _ in range(50):
                model.train_step(memory, batch_size) 
        if returnReward:
            epReward += reward
        if done:
            state = process_state(env.reset())
            done = False
            if returnReward:
                stepList.append(step)
                epochReward.append(epReward)
                epReward = 0
    if returnReward:
        return stepList, epochReward, action
    return step

def train_model(env, memory, model, epIter=200):
    reward_plot = [] # Format of (time step, ep_reward) pairs
    tStep = 0
    for i in range(epIter):
        if ((i+1)%10 == 0): 
            stepList, epochReward, act = runEp(env, memory, model, False, returnReward=True)
            step = stepList[-1]
            for j in range(len(stepList)):
                stepList[j] += tStep
            reward_plot.extend([[a,b] for a,b in zip(stepList, epochReward)])
        else:
            step = runEp(env, memory, model, returnlp=False)
            print("Printing Step")
            print(step)
        if ((i+1)%20 == 0):
            print("Last action on last episode: {}".format(act))
            #print("Runtime (hours) at checkpoint: {}".format((time.time()-start_time)/3600))
            '''
            plt.plot([v[0] for v in reward_plot], [v[1] for v in reward_plot])
            plt.title("Episode Reward vs TimeStep")
            plt.show()
            '''
            model.save()
        tStep += step
    pickle.dump(reward_plot, open('results/DDPG_cheetah.p', 'wb'))

def main():
   
    start_time = time.time()
    domainName = 'walker'
    taskName = 'stand'
    print("Using Test Mode: {}".format(args.test))
    print("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Device Count: {}".format(torch.cuda.device(0)))
        print("Device Name (first): {}".format(torch.cuda.get_device_name(0)))
    print("Domain Used: {}".format(domainName))
    print("Task Used: {}".format(taskName))

    env = suite.load(domain_name=domainName, task_name=taskName)
    obsSpec = env.observation_spec()
    orientDim = obsSpec['orientations'].shape[0]
    heightDim = len(obsSpec['height'].shape) + 1 
    velocityDim = obsSpec['velocity'].shape[0]
    state_space = orientDim + heightDim + velocityDim

    action_space =  env.action_spec().shape[0]

    hp = {'tau': 0.005,\
            'lr': 3*(1e-4),\
            'batch_size': 256,\
            'gamma': 0.99,\
            'max_action': 1.,\
            'min_action': -1.\
            }
    
    model = DDPG(state_space, action_space)

    if args.test is False:
        rmemory = ReplayBuffer(int(1e6))
        train_model(env, rmemory, model, 200)
        # recordEps(env, model, 'vid/DDPGhalfCheetah.mp4')
        print("Baseline comparison")
        # for _ in range(5):
            # baseEp(env)
    # else:
    #     model.load('models/DDPG_cheetah')
    #     if torch.cuda.is_available():
    #         recordEps(env, model, False, 'vid/halfCheetah.mp4')
    #     else:
    #         recordEps(env, model, False, 'vid/halfCheetah.mp4')
    #     for _ in range(5):
    #         baseEp(env)
    # print("End runtime: {} seconds".format(time.time()-start_time))

if __name__ == '__main__':
    main()
