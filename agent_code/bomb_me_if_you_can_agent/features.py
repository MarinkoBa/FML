import numpy as np
import torch
import settings
import scipy.spatial.distance as distance


def features_3x3_space(game_state):
    field = game_state.get('field').T
    own_pos = game_state.get('self')[3]

    up, down, left, right = True, True, True, True
    up_steps, down_steps, left_steps, right_steps = 0, 0, 0, 0
    for i in range(14):
        if up:
            if field[own_pos[1] - i, own_pos[0]] == -1 or field[own_pos[1] - i, own_pos[0]] == 1:
                up = False
            elif field[own_pos[1] - i, own_pos[0]] == 0:
                up_steps = up_steps+1
        if down:
            if field[own_pos[1] + i, own_pos[0]] == -1 or field[own_pos[1] + i, own_pos[0]] == 1:
                down = False
            elif field[own_pos[1] + i, own_pos[0]] == 0:
                down_steps = down_steps+1
        if left:
            if field[own_pos[1], own_pos[0] - i] == -1 or field[own_pos[1], own_pos[0] - i] == 1:
                left = False
            elif field[own_pos[1], own_pos[0] - i] == 0:
                left_steps = left_steps+1
        if right:
            if field[own_pos[1], own_pos[0] + i] == -1 or field[own_pos[1], own_pos[0] + i] == 1:
                right = False
            elif field[own_pos[1], own_pos[0] + i] == 0:
                right_steps = right_steps+1

    actions = np.asarray([[0, up_steps-1, 0], [left_steps-1, 0, right_steps-1], [0, down_steps-1, 0]])
    actions = torch.tensor(actions).double()

    coins = game_state.get('coins')
    field = np.zeros_like(field)

    # get orientation of nearest_coin
    down, up, left, left_up, left_down, right, right_up, right_down = 0, 0, 0, 0, 0, 0, 0, 0

    man_dis = []
    if len(coins) > 0:
        for coin in coins:
            field[coin[1], coin[0]] = 1
            # determine manhattan distance to each coin on field
            man_dis.append(distance.cityblock([own_pos[1], own_pos[0]], [coin[1], coin[0]]))

    nearest_coin_dist = 0

    if len(man_dis) > 0:
        # select coin with shortest distance
        nearest_coin = np.argmin(man_dis)
        nearest_coin_dist = man_dis[nearest_coin]
        nearest_coin_pos = coins[nearest_coin]

        # Determine Orientation of coin
        if own_pos[1] == nearest_coin_pos[1]:
            if own_pos[0] >= nearest_coin_pos[0]:
                left = 1
            else:
                right = 1
        elif own_pos[0] == nearest_coin_pos[0]:
            if own_pos[1] >= nearest_coin_pos[1]:
                up = 1
            else:
                down = 1
        elif own_pos[0] < nearest_coin_pos[0] and own_pos[1] < nearest_coin_pos[1]:
            right_down = 1
        elif own_pos[0] < nearest_coin_pos[0] and own_pos[1] > nearest_coin_pos[1]:
            right_up = 1
        elif own_pos[0] > nearest_coin_pos[0] and own_pos[1] > nearest_coin_pos[1]:
            left_up = 1
        elif own_pos[0] > nearest_coin_pos[0] and own_pos[1] < nearest_coin_pos[1]:
            left_down = 1

    coins = np.asarray([[left_up, up, right_up], [left, 0, right], [left_down, down, right_down]])
    mask = coins == 1
    coins[mask] = nearest_coin_dist

    coins = torch.tensor(coins).double()

    # calculate explosion of bombs
    field = game_state.get('field').T
    bombs = game_state.get('bombs')
    bomb_field = np.zeros_like(field)

    # calculate field with all bomb and expected explosions
    up, down, left, right = True, True, True, True
    if len(bombs) > 0:
        for bomb in bombs:
            bomb_timer = bomb[1]
            bomb_field[bomb[0][1], bomb[0][0]] = bomb_timer

            for i in range(settings.BOMB_POWER + 1):
                if up:
                    if field[bomb[0][1] + i, bomb[0][0]] == -1:
                        up = False
                if down:
                    if field[bomb[0][1] - i, bomb[0][0]] == -1:
                        down = False
                if right:
                    if field[bomb[0][1], bomb[0][0] + i] == -1:
                        right = False
                if left:
                    if field[bomb[0][1], bomb[0][0] - i] == -1:
                        left = False

                if up:
                    bomb_field[bomb[0][1] + i, bomb[0][0]] = bomb_timer
                if down:
                    bomb_field[bomb[0][1] - i, bomb[0][0]] = bomb_timer
                if right:
                    bomb_field[bomb[0][1], bomb[0][0] + i] = bomb_timer
                if left:
                    bomb_field[bomb[0][1], bomb[0][0] - i] = bomb_timer

    # Get 3x3 part from the bomb_field
    down, up, left, left_up, left_down, right, right_up, right_down = 0, 0, 0, 0, 0, 0, 0, 0
    middle = bomb_field[own_pos[1], own_pos[0]]
    down = bomb_field[own_pos[1] + 1, own_pos[0]]
    down_left = bomb_field[own_pos[1] + 1, own_pos[0] - 1]
    down_right = bomb_field[own_pos[1] + 1, own_pos[0] + 1]

    up = bomb_field[own_pos[1] - 1, own_pos[0]]
    up_left = bomb_field[own_pos[1] - 1, own_pos[0] - 1]
    up_right = bomb_field[own_pos[1] - 1, own_pos[0] + 1]

    left = bomb_field[own_pos[1], own_pos[0] - 1]
    right = bomb_field[own_pos[1], own_pos[0] + 1]

    bombs = np.asarray([[up_left, up, up_right], [left, middle, right], [down_left, down, down_right]])
    bombs = torch.tensor(bombs).double()

    # Get 3x3 Info About crates
    field = game_state.get('field').T
    own_pos = game_state.get('self')[3]

    # check if neighboured fields are valid
    # count steps
    up, down, left, right = True, True, True, True
    up_steps, down_steps, left_steps, right_steps = 0, 0, 0, 0
    for i in range(14):
        if up:
            if field[own_pos[1] - i, own_pos[0]] == -1:
                up = False
                up_steps = -1
            elif field[own_pos[1] - i, own_pos[0]] == 1:
                up = False
                up_steps = i
        if down:
            if field[own_pos[1] + i, own_pos[0]] == -1:
                down = False
                down_steps = -1
            elif field[own_pos[1] + i, own_pos[0]] == 1:
                down = False
                down_steps = i
        if left:
            if field[own_pos[1], own_pos[0] - i] == -1:
                left = False
                left_steps = -1
            elif field[own_pos[1], own_pos[0] - i] == 1:
                left = False
                left_steps = i
        if right:
            if field[own_pos[1], own_pos[0] + i] == -1:
                right = False
                right_steps = -1
            elif field[own_pos[1], own_pos[0] + i] == 1:
                right = False
                right_steps = i

    crates = np.asarray([[0, up_steps, 0], [left_steps, 0, right_steps], [0, down_steps, 0]])
    crates = torch.tensor(crates).double()
    #
    # stacked_channels = torch.stack((self_ch_ten, other0_ch_ten, bombs_ch_ten, coins_ch_ten, explosion_map_ch_ten), 0)
    stacked_channels = torch.stack((actions, coins, bombs, crates), 0)
    return stacked_channels
