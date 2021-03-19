import os
import pickle
import random

import numpy as np
import torch
from torch import nn
from agent_code.bomb_me_if_you_can_agent.q_conv_network import Q_Net
import timeit
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

        self.trainings_model = Q_Net()
        self.trainings_model.cuda(0)
        self.trainings_model.train()
        self.trainings_model.double()

        self.optimizer = torch.optim.Adam(params=self.trainings_model.parameters(), lr=0.01)
        self.criterion = nn.SmoothL1Loss()
        self.criterion.cuda(0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.target_model = Q_Net()
        self.target_model.cuda(0)
        self.target_model.double()

        self.actions = constants.ACTIONS
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.trainings_model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    start = timeit.timeit()
    print("start")

    # todo Exploration vs exploitation
    random_prob = constants.EPS
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")

        # 80%: walk in any direction. 10% wait. 10% bomb.
        end = timeit.timeit()
        print(str((end - start) * 1000) + " sec RANDOM")
        # action = np.random.choice(constants.ACTIONS, p=[.2, .2, .2, .2, .2])#, .1])
        # action = np.random.choice(constants.ACTIONS, valid_actions(game_state))

        valid_actions_list = valid_actions(game_state)
        action = np.random.choice(constants.ACTIONS, p=valid_actions_list)
        print(str(action))
        return action

    game_state_features = state_to_features(game_state)
    prediction = self.trainings_model(game_state_features)
    action_pos = torch.argmax(prediction)

    end = timeit.timeit()
    print(str((end - start) * 1000) + " sec MODEL")
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

    self_coord = game_state.get('self')[3]
    self_ch = np.zeros_like(game_state.get('field'))
    self_ch[self_coord[0], self_coord[1]] = 10
    self_ch_ten = torch.from_numpy(self_ch).double()

    others = game_state.get('others')
    other_agents_ch = np.zeros_like(game_state.get('field'))

    if len(others) >= 1:
        other0_coord = others[0][3]
        other_agents_ch[other0_coord[0], other0_coord[1]] = 1

    if len(others) >= 2:
        other1_coord = others[1][3]
        other_agents_ch[other1_coord[0], other1_coord[1]] = 1

    if len(others) >= 3:
        other2_coord = others[2][3]
        other_agents_ch[other2_coord[0], other2_coord[1]] = 1

    other0_ch_ten = torch.from_numpy(other_agents_ch).double()

    bombs = game_state.get('bombs')
    bombs_ch = np.zeros_like(game_state.get('field'))

    if len(bombs) > 0:
        for bomb in bombs:
            bombs_ch[bomb[0][1], bomb[0][0]] = bomb[1]

            x_coor = bomb[0][0]
            y_coor = bomb[0][1]

            x_coords_bomb = [x_coor + 1, x_coor + 2, x_coor + 3, x_coor - 1, x_coor - 2, x_coor - 3]
            y_coords_bomb=[y_coor + 1, y_coor + 2, y_coor + 3, y_coor - 1, y_coor - 2, y_coor - 3]

            for i in range(len(x_coords_bomb)):
                if (x_coords_bomb[i] <= 15 and x_coords_bomb[i] >= 1):
                    bombs_ch[y_coor, x_coords_bomb[i]] = bomb[1]

            for i in range(len(y_coords_bomb)):
                if (y_coords_bomb[i] <= 15 and y_coords_bomb[i] >= 1):
                    bombs_ch[y_coords_bomb[i], x_coor] = bomb[1]

    #bombs_ch_ten = torch.from_numpy(bombs_ch).double()

    coins = game_state.get('coins')
    coins_ch = np.zeros_like(game_state.get('field'))
    if len(coins) > 0:
        for coin in coins:
            coins_ch[coin[0], coin[1]] = 1
    coins_ch_ten = torch.from_numpy(coins_ch).double()

    explosion_map_ch = game_state.get('explosion_map')
    explosion_map_ch_ten = torch.from_numpy(explosion_map_ch).double()

    field_ch = game_state.get('field').transpose()
    field_ch[self_coord[1], self_coord[0]] = 100

    field_ch[field_ch == -1] = 10
    field_ch[field_ch == 0] = 150
    field_ch[field_ch == 1] = 220
    field_ch[coins_ch==1] = 150
    # field_ch[other_agents_ch==1] = 150

    field_ch = field_ch[self_coord[1] - 1:self_coord[1] + 2, self_coord[0] - 1:self_coord[0] + 2]
    coins_ch = coins_ch[self_coord[1] - 1:self_coord[1] + 2, self_coord[0] - 1:self_coord[0] + 2]
    bombs_ch = bombs_ch[self_coord[1] - 1:self_coord[1] + 2, self_coord[0] - 1:self_coord[0] + 2]

    bombs_ch_ten = torch.from_numpy(bombs_ch).double()

    min_dist = 1000
    x_dist_min = 0
    y_dist_min = 0
    coin_min = tuple()
    for coin in coins:
        dist = np.sqrt(np.power(self_coord[0] - coin[0], 2) + np.power(self_coord[1] - coin[1], 2))
        if (dist < min_dist):
            min_dist = dist
            x_dist_min = coin[0] - self_coord[0]
            y_dist_min = self_coord[1] - coin[1]
            coin_min = coin

    if(min_dist==0):
        min_dist=1
    coin_dist = 150 + int((105/min_dist))

    if (x_dist_min > 0 and y_dist_min > 0):
        field_ch[0][2] = coin_dist
    elif (x_dist_min > 0 and y_dist_min == 0):
        field_ch[1][2] = coin_dist
    elif (x_dist_min > 0 and y_dist_min < 0):
        field_ch[2][2] = coin_dist

    elif (x_dist_min == 0 and y_dist_min < 0):
        field_ch[2][1] = coin_dist
    elif (x_dist_min == 0 and y_dist_min > 0):
        field_ch[0][1] = coin_dist

    elif (x_dist_min < 0 and y_dist_min > 0):
        field_ch[0][0] = coin_dist
    elif (x_dist_min < 0 and y_dist_min == 0):
        field_ch[1][0] = coin_dist
    elif (x_dist_min < 0 and y_dist_min < 0):
        field_ch[2][0] = coin_dist

    field_ch_ten = torch.from_numpy(field_ch).double()

    stacked_channels = torch.stack((field_ch_ten, bombs_ch_ten), 0)
    stacked_channels = stacked_channels.unsqueeze(0)

    return stacked_channels


def valid_actions(game_state):
    field = game_state.get('field')
    own_pos = game_state.get('self')[3]

    # check if neighboured fields are valid
    up = field[own_pos[1] + 1, own_pos[0]] == 0
    down = field[own_pos[1] - 1, own_pos[0]] == 0
    left = field[own_pos[1], own_pos[0] - 1] == 0
    right = field[own_pos[1], own_pos[0] + 1] == 0

    count = np.sum([up, right, down, left])

    if(count==0):
        return [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    return [0.8 * up / count, 0.8 * right / count, 0.8 * down / count, 0.8 * left / count, 0.1 , 0.1]
