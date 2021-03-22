import os
import pickle
import random

import operator

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
    if self.train or not os.path.isfile("2crate_destr_sch_cor_place_bomb_other_dir_upscal_no_gui_lr0001.pt"):
        self.logger.info("Setting up model from scratch.")

        self.trainings_model = Q_Net()
        self.trainings_model.cuda(0)
        self.trainings_model.train()
        self.trainings_model.double()

        self.optimizer = torch.optim.Adam(params=self.trainings_model.parameters(), lr=0.0001)
        self.criterion = nn.SmoothL1Loss()
        self.criterion.cuda(0)

        self.target_model = Q_Net()
        self.target_model.cuda(0)
        self.target_model.double()

        self.actions = constants.ACTIONS
    else:
        self.logger.info("Loading model from saved state.")
        with open("5crate_destr_cor_place_bomb_2step_where_upscal_no_gui_lr0001.pt", "rb") as file:
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

    # Current features 3->3x3
    # 1. Field 150 weg, 10 wand, >150 crate, >150coin
    # 2. Bomb 255 if there is no bomb danger, 3,2,1 if there is danger
    # 3. Free dir -> slice two upper rows, two bottom rows, two left columns and two right columns in 3x3 and check where is
    # more place to walk (more ways)

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
    bombs_ch = np.ones_like(game_state.get('field')) * 255

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

    for bomb in bombs:
        field_ch[bomb[0][1], bomb[0][0]] = 2

    #recursive_neighborhood(self_coord, field_ch)

    res = find_3neighbor_values(self_coord, field_ch)

    free_dir = np.ones((3,3))*10
    free_dir[0][1] = res[0]
    free_dir[2][1] = res[1]
    free_dir[1][0] = res[2]
    free_dir[1][2] = res[3]


    #
    # free_dir = np.zeros((3,3))
    # cnt_free_steps = 0
    # up = (self_coord[0], self_coord[1] - 1)
    # down = (self_coord[0], self_coord[1] + 1)
    # left = (self_coord[0] - 1, self_coord[1])
    # right = (self_coord[0] + 1, self_coord[1])
    # step_arr = [up, down, left, right]
    # for ind, st in enumerate(step_arr):
    #     cnt_free_steps = 0
    #     up2 = st[0], st[1] - 1
    #     down2 = st[0], st[1] + 1
    #     left2 = st[0] - 1, st[1]
    #     right2 = st[0] + 1, st[1]
    #     if (field_ch[st[1],st[0]] == 0):
    #         cnt_free_steps += 1
    #         if (ind != 1 and up2[0] >= 1 and up2[0] <= 15 and up2[1] >= 1 and up2[1] <= 15):
    #             if (field_ch[up2[1],up2[0]] == 0):
    #                 cnt_free_steps += 1
    #         if (ind != 0 and down2[0] >= 1 and down2[0] <= 15 and down2[1] >= 1 and down2[1] <= 15):
    #             if (field_ch[down2[1],down2[0]] == 0):
    #                 cnt_free_steps += 1
    #         if (ind != 3 and left2[0] >= 1 and left2[0] <= 15 and left2[1] >= 1 and left2[1] <= 15):
    #             if (field_ch[left2[1],left2[0]] == 0):
    #                 cnt_free_steps += 1
    #         if (ind != 2 and right2[0] >= 1 and right2[0] <= 15 and right2[1] >= 1 and right2[1] <= 15):
    #             if (field_ch[right2[1],right2[0]] == 0):
    #                 cnt_free_steps += 1
    #     if(ind==0):
    #         free_dir[0][1]=cnt_free_steps * 100
    #     elif (ind == 1):
    #         free_dir[2][1] = cnt_free_steps * 100
    #     elif (ind == 2):
    #         free_dir[1][0] = cnt_free_steps * 100
    #     elif (ind == 3):
    #         free_dir[1][2] = cnt_free_steps * 100
    #
    # free_dir[free_dir==0] = 10

    field_ch[self_coord[1], self_coord[0]] = 100

    field_ch[field_ch == -1] = 10
    field_ch[field_ch == 0] = 150
    field_ch[field_ch == 1] = 220
    field_ch[coins_ch==1] = 150

    for bomb in bombs:
        field_ch[bomb[0][1], bomb[0][0]] = bomb[1]
    # field_ch[other_agents_ch==1] = 150

    ii = np.where(field_ch == 220)
    crate_coords = np.stack((ii[1], ii[0]))


    field_ch = field_ch[self_coord[1] - 1:self_coord[1] + 2, self_coord[0] - 1:self_coord[0] + 2]
    coins_ch = coins_ch[self_coord[1] - 1:self_coord[1] + 2, self_coord[0] - 1:self_coord[0] + 2]
    bombs_ch = bombs_ch[self_coord[1] - 1:self_coord[1] + 2, self_coord[0] - 1:self_coord[0] + 2]

    bombs_ch_ten = torch.from_numpy(bombs_ch).double()



    min_dist_crate = 1000
    x_dist_min_crate = 0
    y_dist_min_crate = 0
    for crate in range(len(crate_coords[1])):
        dist = np.sqrt(np.power(self_coord[0] - crate_coords[0][crate], 2) + np.power(self_coord[1] -  crate_coords[1][crate], 2))
        if (dist < min_dist_crate):
            min_dist_crate = dist
            x_dist_min_crate = crate_coords[0][crate] - self_coord[0]
            y_dist_min_crate = self_coord[1] - crate_coords[1][crate]
            coin_min = crate

    if (min_dist_crate == 0):
        min_dist_crate = 1
    crate_dist = 150 + int((105 / min_dist_crate))

    if (x_dist_min_crate > 0 and y_dist_min_crate > 0):
        field_ch[0][2] = crate_dist
    elif (x_dist_min_crate > 0 and y_dist_min_crate == 0):
        field_ch[1][2] = crate_dist
    elif (x_dist_min_crate > 0 and y_dist_min_crate < 0):
        field_ch[2][2] = crate_dist

    elif (x_dist_min_crate == 0 and y_dist_min_crate < 0):
        field_ch[2][1] = crate_dist
    elif (x_dist_min_crate == 0 and y_dist_min_crate > 0):
        field_ch[0][1] = crate_dist

    elif (x_dist_min_crate < 0 and y_dist_min_crate > 0):
        field_ch[0][0] = crate_dist
    elif (x_dist_min_crate < 0 and y_dist_min_crate == 0):
        field_ch[1][0] = crate_dist
    elif (x_dist_min_crate < 0 and y_dist_min_crate < 0):
        field_ch[2][0] = crate_dist

    field_ch[field_ch == 220] = crate_dist

    min_dist = 1000
    x_dist_min = 0
    y_dist_min = 0
    for coin in coins:
        dist = np.sqrt(np.power(self_coord[0] - coin[0], 2) + np.power(self_coord[1] - coin[1], 2))
        if (dist < min_dist):
            min_dist = dist
            x_dist_min = coin[0] - self_coord[0]
            y_dist_min = self_coord[1] - coin[1]

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

    free_dir_ten = torch.from_numpy(free_dir).double()
    field_ch_ten = torch.from_numpy(field_ch).double()

    stacked_channels = torch.stack((field_ch_ten, bombs_ch_ten, free_dir_ten), 0)
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


def find_3neighbor_values(self_coord, field_ch):
    up = (self_coord[0], self_coord[1] - 1)

    up_up = (up[0], up[1] - 1)
    up_left = (up[0] - 1, up[1])
    up_right = (up[0] + 1, up[1])

    up_up_up = (up_up[0], up_up[1] - 1)
    up_up_left = (up_up[0] - 1, up_up[1])
    up_up_right = (up_up[0] + 1, up_up[1])

    up_left_up = (up_left[0], up_left[1] - 1)
    up_left_down = (up_left[0], up_left[1] + 1)
    up_left_left = (up_left[0] - 1, up_left[1])

    up_right_up = (up_right[0], up_right[1] - 1)
    up_right_down = (up_right[0], up_right[1] + 1)
    up_right_right = (up_right[0] + 1, up_right[1])

    up_list_coord = [up,
               up_up, up_left, up_right,
               up_up_up, up_up_left, up_up_right,
               up_left_up, up_left_down, up_left_left,
               up_right_up, up_right_down, up_right_right]

    up_list = np.ones(len(up_list_coord)) * 5

    if(field_ch[up_list_coord[0][1], up_list_coord[0][0]] != 0):
        up_list = np.ones(len(up_list_coord))
    else:
        if(field_ch[up_list_coord[1][1], up_list_coord[1][0]] != 0):
            up_list[1]=1
            up_list[4]=1
            up_list[5]=1
            up_list[6]=1
        if (field_ch[up_list_coord[2][1], up_list_coord[2][0]] != 0):
            up_list[2] = 1
            up_list[7] = 1
            up_list[8] = 1
            up_list[9] = 1
        if (field_ch[up_list_coord[3][1], up_list_coord[3][0]] != 0):
            up_list[3] = 1
            up_list[10] = 1
            up_list[11] = 1
            up_list[12] = 1


    down = (self_coord[0], self_coord[1] + 1)

    down_down = (down[0], down[1] + 1)
    down_left = (down[0] - 1, down[1])
    down_right = (down[0] + 1, down[1])

    down_down_down = (down_down[0], down_down[1] + 1)
    down_down_left = (down_down[0] - 1, down_down[1])
    down_down_right = (down_down[0] + 1, down_down[1])

    down_left_up = (down_left[0], down_left[1] - 1)
    down_left_down = (down_left[0], down_left[1] + 1)
    down_left_left = (down_left[0] - 1, down_left[1])

    down_right_up = (down_right[0], down_right[1] - 1)
    down_right_down = (down_right[0], down_right[1] + 1)
    down_right_right = (down_right[0] + 1, down_right[1])

    down_list_coord = [down,
               down_down, down_left, down_right,
               down_down_down, down_down_left, down_down_right,
               down_left_up, down_left_down, down_left_left,
               down_right_up, down_right_down, down_right_right]

    down_list = np.ones(len(down_list_coord)) * 5

    if(field_ch[down_list_coord[0][1], down_list_coord[0][0]] != 0):
        down_list = np.ones(len(down_list_coord))
    else:
        if(field_ch[down_list_coord[1][1], down_list_coord[1][0]] != 0):

            down_list[1] = 1
            down_list[4] = 1
            down_list[5] = 1
            down_list[6] = 1
        if (field_ch[down_list_coord[2][1], down_list_coord[2][0]] != 0):
            down_list[2] = 1
            down_list[7] = 1
            down_list[8] = 1
            down_list[9] = 1
        if (field_ch[down_list_coord[3][1], down_list_coord[3][0]] != 0):
            down_list[3] = 1
            down_list[10] = 1
            down_list[11] = 1
            down_list[12] = 1



    left = (self_coord[0] - 1, self_coord[1])

    left_down = (left[0], left[1] + 1)
    left_left = (left[0] - 1, left[1])
    left_up = (left[0], left[1] - 1)

    left_down_down = (left_down[0], left_down[1] + 1)
    left_down_left = (left_down[0] - 1, left_down[1])
    left_down_right = (left_down[0] + 1, left_down[1])

    left_left_up = (left_left[0], left_left[1] - 1)
    left_left_down = (left_left[0], left_left[1] + 1)
    left_left_left = (left_left[0] - 1, left_left[1])

    left_up_up = (left_up[0], left_up[1] - 1)
    left_up_left = (left_up[0] - 1, left_up[1])
    left_up_right = (left_up[0] + 1, left_up[1])

    left_list_coord = [left,
                 left_down, left_left, left_up,
                 left_down_down, left_down_left, left_down_right,
                 left_left_up, left_left_down, left_left_left,
                 left_up_up, left_up_left, left_up_right]


    left_list = np.ones(len(left_list_coord)) * 5

    if(field_ch[left_list_coord[0][1], left_list_coord[0][0]] != 0):
        left_list = np.ones(len(left_list_coord))
    else:
        if(field_ch[left_list_coord[1][1], left_list_coord[1][0]] != 0):
            left_list[1] = 1
            left_list[4] = 1
            left_list[5] = 1
            left_list[6] = 1
        if (field_ch[left_list_coord[2][1], left_list_coord[2][0]] != 0):
            left_list[2] = 1
            left_list[7] = 1
            left_list[8] = 1
            left_list[9] = 1
        if (field_ch[left_list_coord[3][1], left_list_coord[3][0]] != 0):
            left_list[3] = 1
            left_list[10] = 1
            left_list[11] = 1
            left_list[12] = 1

    right = (self_coord[0] + 1, self_coord[1])

    right_down = (right[0], right[1] + 1)
    right_right = (right[0] + 1, right[1])
    right_up = (right[0], right[1] - 1)

    right_down_down = (right_down[0], right_down[1] + 1)
    right_down_left = (right_down[0] - 1, right_down[1])
    right_down_right = (right_down[0] + 1, right_down[1])

    right_right_up = (right_right[0], right_right[1] - 1)
    right_right_down = (right_right[0], right_right[1] + 1)
    right_right_right = (right_right[0] + 1, right_right[1])

    right_up_up = (right_up[0], right_up[1] - 1)
    right_up_left = (right_up[0] - 1, right_up[1])
    right_up_right = (right_up[0] + 1, right_up[1])


    right_list_coord = [right,
                 right_down, right_right, right_up,
                 right_down_down, right_down_left, right_down_right,
                 right_right_up, right_right_down, right_right_right,
                 right_up_up, right_up_left, right_up_right]


    right_list = np.ones(len(right_list_coord)) * 5

    if(field_ch[right_list_coord[0][1], right_list_coord[0][0]] != 0):
        right_list = np.ones(len(right_list_coord))
    else:
        if(field_ch[right_list_coord[1][1], right_list_coord[1][0]] != 0):
            right_list[1] = 1
            right_list[4] = 1
            right_list[5] = 1
            right_list[6] = 1
        if (field_ch[right_list_coord[2][1], right_list_coord[2][0]] != 0):
            right_list[2] = 1
            right_list[7] = 1
            right_list[8] = 1
            right_list[9] = 1
        if (field_ch[right_list_coord[3][1], right_list_coord[3][0]] != 0):
            right_list[3] = 1
            right_list[10] = 1
            right_list[11] = 1
            right_list[12] = 1

    duplicates_up_list = [idx for idx, val in enumerate(up_list_coord) if val in up_list_coord[:idx]]
    duplicates_down_list = [idx for idx, val in enumerate(down_list_coord) if val in down_list_coord[:idx]]
    duplicates_left_list = [idx for idx, val in enumerate(left_list_coord) if val in left_list_coord[:idx]]
    duplicates_right_list = [idx for idx, val in enumerate(right_list_coord) if val in right_list_coord[:idx]]

    for i in range(len(duplicates_up_list)):
        up_list[duplicates_up_list[i]] = 1

    for j in range(len(duplicates_down_list)):
        down_list[duplicates_down_list[j]] = 1

    for k in range(len(duplicates_left_list)):
        left_list[duplicates_left_list[k]] = 1

    for l in range(len(duplicates_right_list)):
        right_list[duplicates_right_list[l]] = 1


    list_of_lists_coord = [up_list_coord, down_list_coord, left_list_coord, right_list_coord]
    list_of_lists_val = [up_list, down_list, left_list, right_list]

    final_res = []

    for i in range(len(list_of_lists_coord)):
        counter = 0
        for j in range(len(list_of_lists_coord[i])):
            if((list_of_lists_coord[i][j][1] >= 1 and list_of_lists_coord[i][j][1] <= 15 and list_of_lists_coord[i][j][0] >= 1 and list_of_lists_coord[i][j][0] <= 15) and (field_ch[list_of_lists_coord[i][j][1],list_of_lists_coord[i][j][0]] == 0) and (list_of_lists_val[i][j] == 5)):
                counter+=1
        final_res.append(counter * 100)

    return final_res




