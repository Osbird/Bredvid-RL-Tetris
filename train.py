"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque
import cv2

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--render", type=bool, default=True, help="Render video? Much faster training without rendering, and more educational with rendering")

    args = parser.parse_args()
    return args


def train(opt):
    cv2.setUseOptimized(True)
    print("cv2 is optimized =", cv2.useOptimized())
    print("cuda available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    
    #############
    # The environment in this situation is the Tetris game itself
    # Create a Tetris object with the constructor in src/tetris.py and pass in the 3 arguments described there
    # the arguments default values are stored in "opt" which comes from the argument parser above 
    #############
    env = #.....
    

    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    
    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        
        #############
        # epsilon = A threshhold that decides how likely it is that a random action is performed. also often calles "eps_threshold"
        # Insert a function that creates a decaying value per epoch, between 1 and 0. amount of epochs is defined as input, and default in parser(top of this file)
        # The best results will probably come from a function that decays in the first x epochs and is = "some final low value, e.g. 0.001" for the last y epochs
        #############
        epsilon = #.....       
        
        #############
        # u = a random number between 0 and 1 that randomly decides whether a random action is performed or an action chosen by the model
        #############
        u = #.....
        random_action = u <= epsilon 

    
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        
        # if cuda is available, move network to GPU 
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        
        model.#......      evaluate model, built in function
        
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        #############
        # reward = metric of how well the model has performed
        # done = whether or not the model is finished with the current game 
        # a function that outputs both of these state variables can be found in src/tetris.py
        # remeber to pass in the second argument as well, otherwise all training will be done with visualization which is cool to watch, but slow
        #############
        reward, done = env.#.....

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        
        #..... Increment epoch counter

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        # if cuda is available, move network to GPU 
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)

        model.#......      evaluate model, built in function

        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        
        #############
        # Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses
        # 
        # In order to ensure that behaviour learned from one epoch is not used in multiple epochs to tune a network(accumulating gradients), 
        # we need to set all gradients to zero, this is a built in function 
        #############
        optimizer.#.....
        
        #############
        # A criterion is a loss function that compares two inputs as tensors(pytorch datatype)
        # 
        # Pass in two tensors into our criterion(loss function, set to MSELoss())
        #############
        loss = criterion()#(tensorX, tensorY)
        
        #############
        # The criterion outputs a loss object that stores the loss value and enables us to tune our neural network by backpropagation
        # 
        # The loss function has access to a built in function that does this
        #############
        loss.#.....

        #############
        # Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses
        # 
        # The optimizer should her e perform a parameter update based on the current gradient, and does this by a built in function
        #############
        optimizer.#.....

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    #############
    # PyTorch offers a way of saving either full models or a model state's weights as a file
    # 
    # Search on pytorch.org for this save function
    #############
    torch.#.....


if __name__ == "__main__":
    opt = get_args()
    train(opt)
