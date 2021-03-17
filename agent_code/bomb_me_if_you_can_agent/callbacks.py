import os
import pickle
import random

import numpy as np
import torch
from torch import nn
from agent_code.bomb_me_if_you_can_agent.network import Q_Net
import time
from agent_code.bomb_me_if_you_can_agent import constants


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

        self.policy_model = Q_Net()
        self.policy_model.cuda(0)
        self.policy_model.train()
        self.policy_model.double()

        self.optimizer = torch.optim.Adam(params=self.policy_model.parameters(), lr=0.01)
        self.criterion = nn.SmoothL1Loss()
        self.criterion.cuda(0)

        self.target_model = Q_Net()
        self.target_model.cuda()
        self.target_model.double()
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.actions = constants.ACTIONS
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    start = time.time()
    print("")

    # todo Exploration vs exploitation
    random_prob = constants.EPS
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")

        # 80%: walk in any direction. 10% wait. 10% bomb.
        end = time.time()
        print(str((end - start)) + " sec RANDOM")
        valid_actions_list = valid_actions(game_state)


        action = np.random.choice(constants.ACTIONS, p=valid_actions_list)

        print(str(action))
        return action

    game_state_features = state_to_features(game_state)
    prediction = self.target_model(game_state_features)
    action_pos = torch.argmax(prediction)

    end = time.time()
    print(str((end - start)) + " sec MODEL")
    print(str(constants.ACTIONS[action_pos]))
    return constants.ACTIONS[action_pos]


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

    # field_ch = game_state.get('field')
    # field_ch_ten = torch.from_numpy(field_ch).double()
    #
    # self_coord = game_state.get('self')[3]
    # self_ch = np.zeros_like(game_state.get('field'))
    # self_ch[self_coord[0], self_coord[1]] = 1
    # self_ch_ten = torch.from_numpy(self_ch).double()
    #
    # others = game_state.get('others')
    # other_agents_ch = np.zeros_like(game_state.get('field'))
    #
    # if len(others) >= 1:
    #     other0_coord = others[0][3]
    #     other_agents_ch[other0_coord[0], other0_coord[1]] = 1
    #
    # if len(others) >= 2:
    #     other1_coord = others[1][3]
    #     other_agents_ch[other1_coord[0], other1_coord[1]] = 1
    #
    # if len(others) >= 3:
    #     other2_coord = others[2][3]
    #     other_agents_ch[other2_coord[0], other2_coord[1]] = 1
    #
    # other0_ch_ten = torch.from_numpy(other_agents_ch).double()
    #
    # bombs = game_state.get('bombs')
    # bombs_ch = np.zeros_like(game_state.get('field'))
    # if len(bombs) > 0:
    #     for bomb in bombs:
    #         bombs_ch[bomb[0], bomb[1]] = 1
    # bombs_ch_ten = torch.from_numpy(bombs_ch).double()
    #
    # coins = game_state.get('coins')
    # coins_ch = np.zeros_like(game_state.get('field'))
    # if len(coins) > 0:
    #     for coin in coins:
    #         coins_ch[coin[0], coin[1]] = 1
    # coins_ch_ten = torch.from_numpy(coins_ch).double()
    #
    # explosion_map_ch = game_state.get('explosion_map')
    # explosion_map_ch_ten = torch.from_numpy(explosion_map_ch).double()
    field_ch = game_state.get('field')

    self_coord = game_state.get('self')[3]
    field_ch[self_coord[0], self_coord[1]] = 2

    others = game_state.get('others')

    if len(others) >= 1:
        other0_coord = others[0][3]
        field_ch[other0_coord[0], other0_coord[1]] = 3

    if len(others) >= 2:
        other1_coord = others[1][3]
        field_ch[other1_coord[0], other1_coord[1]] = 3

    if len(others) >= 3:
        other2_coord = others[2][3]
        field_ch[other2_coord[0], other2_coord[1]] = 3

    bombs = game_state.get('bombs')
    if len(bombs) > 0:
        for bomb in bombs:
            field_ch[bomb[0], bomb[1]] = 4

    coins = game_state.get('coins')
    if len(coins) > 0:
        for coin in coins:
            field_ch[coin[0], coin[1]] = 5

    field_ch_ten = torch.from_numpy(field_ch).double()

    explosion_map_ch = game_state.get('explosion_map')
    explosion_map_ch_ten = torch.from_numpy(explosion_map_ch).double()

    # stacked_channels = torch.stack((self_ch_ten, other0_ch_ten, bombs_ch_ten, coins_ch_ten, explosion_map_ch_ten), 0)
    stacked_channels = torch.stack((field_ch_ten, explosion_map_ch_ten), 0)
    stacked_channels = stacked_channels.unsqueeze(0)
    # stacked_channels = stacked_channels.unsqueeze(0)

    # and return them as a vector
    return stacked_channels


def valid_actions(game_state):
    field = game_state.get('field')
    own_pos = game_state.get('self')[3]

    # check if neighboured fields are valid
    up = field[own_pos[0] + 1, own_pos[0]] == 0
    down = field[own_pos[0] - 1, own_pos[0]] == 0
    left = field[own_pos[0], own_pos[0] - 1] == 0
    right = field[own_pos[0], own_pos[0] + 1] == 0

    count = np.sum([up,right,down,left])

    return [0.8*up/count, 0.8*right/count, 0.8*down/count, 0.8*left/count, 0.1, 0.1]
