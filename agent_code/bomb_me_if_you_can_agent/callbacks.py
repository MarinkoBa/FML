import os
import pickle
import random

import numpy as np
import torch
from torch import nn
from agent_code.bomb_me_if_you_can_agent.network import Q_Net

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    # TODO Model prediction
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []

    #############
    self_coord = game_state.get('self')[3]
    self_ch = np.zeros_like(game_state.get('field'))
    self_ch[self_coord[0], self_coord[1]] = 1
    self_ch_ten = torch.from_numpy(self_ch).double()
    #self_ch_ten = self_ch_ten.reshape(-1)
    #channels.append(self_ch_ten)

    others = game_state.get('others')

    other0_coord = others[0][3]
    other0_ch = np.zeros_like(game_state.get('field'))
    other0_ch[other0_coord[0], other0_coord[1]] = 1
    other0_ch_ten = torch.from_numpy(other0_ch).double()
    #other0_ch_ten = other0_ch_ten.reshape(-1)
    #channels.append(other0_ch_ten)

    other1_coord = others[1][3]
    other1_ch = np.zeros_like(game_state.get('field'))
    other1_ch[other1_coord[0], other1_coord[1]] = 1
    other1_ch_ten = torch.from_numpy(other1_ch).double()
    #other1_ch_ten=other1_ch_ten.reshape(-1)
    #channels.append(other1_ch_ten)

    other2_coord = others[2][3]
    other2_ch = np.zeros_like(game_state.get('field'))
    other2_ch[other2_coord[0], other2_coord[1]] = 1
    other2_ch_ten = torch.from_numpy(other2_ch).double()
    #other2_ch_ten = other2_ch_ten.reshape(-1)
    #channels.append(other2_ch_ten)

    bombs = game_state.get('bombs')
    bombs_ch = np.zeros_like(game_state.get('field'))
    if len(bombs) > 0:
        for bomb in bombs:
            bombs_ch[bomb[0], bomb[1]] = 1
    bombs_ch_ten = torch.from_numpy(bombs_ch).double()

    #bombs_ch_ten = bombs_ch_ten.reshape(-1)
    #channels.append(bombs_ch_ten)

    coins = game_state.get('coins')
    coins_ch = np.zeros_like(game_state.get('field'))
    if len(coins) > 0:
        for coin in coins:
            coins_ch[coin[0], coin[1]] = 1
    coins_ch_ten = torch.from_numpy(coins_ch).double()
    #coins_ch_ten = coins_ch_ten.reshape(-1)
    #channels.append(coins_ch_ten)

    explosion_map_ch = game_state.get('explosion_map')
    explosion_map_ch_ten = torch.from_numpy(explosion_map_ch).double()
    #explosion_map_ch_ten = explosion_map_ch_ten.reshape(-1)
    #channels.append(explosion_map_ch_ten)
    #########

    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)

    tensor_ch = torch.stack(
        (self_ch_ten, other0_ch_ten, other1_ch_ten, other2_ch_ten, bombs_ch_ten, coins_ch_ten, explosion_map_ch_ten), 0)

    tensor_ch = tensor_ch.unsqueeze(0)

    # and return them as a vector
    return tensor_ch
