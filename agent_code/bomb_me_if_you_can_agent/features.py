import numpy as np
import torch
import settings
import scipy.spatial.distance as distance


def features_3x3_space(game_state):
    field = game_state.get('field').T
    own_pos = game_state.get('self')[3]

    # up, down, left, right = True, True, True, True
    # up_steps, down_steps, left_steps, right_steps = 0, 0, 0, 0
    # for i in range(14):
    #     if up:
    #         if field[own_pos[1] - i, own_pos[0]] == -1 or field[own_pos[1] - i, own_pos[0]] == 1:
    #             up = False
    #         elif field[own_pos[1] - i, own_pos[0]] == 0:
    #             up_steps = up_steps + 1
    #     if down:
    #         if field[own_pos[1] + i, own_pos[0]] == -1 or field[own_pos[1] + i, own_pos[0]] == 1:
    #             down = False
    #         elif field[own_pos[1] + i, own_pos[0]] == 0:
    #             down_steps = down_steps + 1
    #     if left:
    #         if field[own_pos[1], own_pos[0] - i] == -1 or field[own_pos[1], own_pos[0] - i] == 1:
    #             left = False
    #         elif field[own_pos[1], own_pos[0] - i] == 0:
    #             left_steps = left_steps + 1
    #     if right:
    #         if field[own_pos[1], own_pos[0] + i] == -1 or field[own_pos[1], own_pos[0] + i] == 1:
    #             right = False
    #         elif field[own_pos[1], own_pos[0] + i] == 0:
    #             right_steps = right_steps + 1
    #
    # actions = np.asarray([[0, up_steps - 1, 0], [left_steps - 1, 0, right_steps - 1], [0, down_steps - 1, 0]])
    # actions = torch.tensor(actions).double()

    free_dir = np.zeros((3, 3))

    bombs = game_state.get('bombs')
    if len(bombs) > 0:
        for bomb in bombs:
            field[bomb[0][1], bomb[0][0]] = 2
    cnt_free_steps = 0

    up = (own_pos[0], own_pos[1] - 1)
    down = (own_pos[0], own_pos[1] + 1)
    left = (own_pos[0] - 1, own_pos[1])
    right = (own_pos[0] + 1, own_pos[1])
    step_arr = [up, down, left, right]
    for ind, st in enumerate(step_arr):
        cnt_free_steps = 0
        up2 = st[0], st[1] - 1
        down2 = st[0], st[1] + 1
        left2 = st[0] - 1, st[1]
        right2 = st[0] + 1, st[1]
        if (field[st[1], st[0]] == 0):
            cnt_free_steps += 1
            if (ind != 1 and up2[0] >= 1 and up2[0] <= 15 and up2[1] >= 1 and up2[1] <= 15):
                if (field[up2[1], up2[0]] == 0):
                    cnt_free_steps += 1
            if (ind != 0 and down2[0] >= 1 and down2[0] <= 15 and down2[1] >= 1 and down2[1] <= 15):
                if (field[down2[1], down2[0]] == 0):
                    cnt_free_steps += 1
            if (ind != 3 and left2[0] >= 1 and left2[0] <= 15 and left2[1] >= 1 and left2[1] <= 15):
                if (field[left2[1], left2[0]] == 0):
                    cnt_free_steps += 1
            if (ind != 2 and right2[0] >= 1 and right2[0] <= 15 and right2[1] >= 1 and right2[1] <= 15):
                if (field[right2[1], right2[0]] == 0):
                    cnt_free_steps += 1
        if (ind == 0):
            free_dir[0][1] = cnt_free_steps
        elif (ind == 1):
            free_dir[2][1] = cnt_free_steps
        elif (ind == 2):
            free_dir[1][0] = cnt_free_steps
        elif (ind == 3):
            free_dir[1][2] = cnt_free_steps

    free_dir = torch.tensor(free_dir).double()

    # initialize field new
    field = game_state.get('field').T
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
                # up_steps = -1
            elif field[own_pos[1] - i, own_pos[0]] == 1:
                up = False
                up_steps = i
        if down:
            if field[own_pos[1] + i, own_pos[0]] == -1:
                down = False
                # down_steps = -1
            elif field[own_pos[1] + i, own_pos[0]] == 1:
                down = False
                down_steps = i
        if left:
            if field[own_pos[1], own_pos[0] - i] == -1:
                left = False
                # left_steps = -1
            elif field[own_pos[1], own_pos[0] - i] == 1:
                left = False
                left_steps = i
        if right:
            if field[own_pos[1], own_pos[0] + i] == -1:
                right = False
                # right_steps = -1
            elif field[own_pos[1], own_pos[0] + i] == 1:
                right = False
                right_steps = i

    crates = np.asarray([[0, up_steps, 0], [left_steps, 0, right_steps], [0, down_steps, 0]])
    mask = crates == 0
    crates[mask] = 100
    nearest_crate_index = np.unravel_index(crates.argmin(), crates.shape)
    nearest_crates = np.zeros_like(crates)
    nearest_crates[nearest_crate_index[0], nearest_crate_index[1]] = crates[
        nearest_crate_index[0], nearest_crate_index[1]]
    crates = torch.tensor(nearest_crates).double()
    #
    # stacked_channels = torch.stack((self_ch_ten, other0_ch_ten, bombs_ch_ten, coins_ch_ten, explosion_map_ch_ten), 0)
    stacked_channels = torch.stack((free_dir, coins, bombs, crates), 0)
    return stacked_channels


def feature_field(game_state):
    field = game_state.get('field').T

    self_coord = game_state.get('self')[3]
    self = np.zeros_like(field)
    self[self_coord[1], self_coord[0]] = 1

    others = game_state.get('others')
    other_field = np.zeros_like(field)
    if len(others) > 0:
        other_coord = others[0][3]
        other_field[other_coord[1], other_coord[0]] = 1

        if len(others) > 1:
            other_coord = others[1][3]
            other_field[other_coord[1], other_coord[0]] = 1
        if len(others) > 2:
            other_coord = others[2][3]
            other_field[other_coord[1], other_coord[0]] = 1

    bombs = game_state.get('bombs')
    bomb_field = np.zeros_like(field)
    if len(bombs) > 0:
        for bomb in bombs:
            bomb_field[bomb[1], bomb[0]] = 1

    coins_coord = game_state.get('coins')
    coins = np.zeros_like(field)
    if len(coins_coord) > 0:
        for coin in coins_coord:
            coins[coin[1], coin[0]] = distance.cityblock([self_coord[1], self_coord[0]], [coin[1], coin[0]])

    # calculate explosion of bombs
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
                    if field[bomb[0][1] + i, bomb[0][0]] == -1 or field[bomb[0][1] + i, bomb[0][0]] == 1:
                        up = False
                if down:
                    if field[bomb[0][1] - i, bomb[0][0]] == -1 or field[bomb[0][1] - i, bomb[0][0]] == 1:
                        down = False
                if right:
                    if field[bomb[0][1], bomb[0][0] + i] == -1 or field[bomb[0][1], bomb[0][0] + i] == 1:
                        right = False
                if left:
                    if field[bomb[0][1], bomb[0][0] - i] == -1 or field[bomb[0][1], bomb[0][0] - i] == 1:
                        left = False

                if up:
                    bomb_field[bomb[0][1] + i, bomb[0][0]] = bomb_timer
                if down:
                    bomb_field[bomb[0][1] - i, bomb[0][0]] = bomb_timer
                if right:
                    bomb_field[bomb[0][1], bomb[0][0] + i] = bomb_timer
                if left:
                    bomb_field[bomb[0][1], bomb[0][0] - i] = bomb_timer

    field = get_7x7_submatrix(field,self_coord)
    others = get_7x7_submatrix(other_field,self_coord)
    bombs = get_7x7_submatrix(bomb_field,self_coord)
    coins = get_7x7_submatrix(coins,self_coord)


    field = torch.tensor(field).double()
    #self = torch.tensor(self).double()
    others = torch.tensor(others).double()
    bombs = torch.tensor(bombs).double()
    coins = torch.tensor(coins).double()

    field = torch.tensor(field).double()
    tensor_ch = torch.stack(
        (field, others, bombs, coins), 0)

    tensor_ch = tensor_ch.unsqueeze(0)

    return tensor_ch

def get_7x7_submatrix(field,self_coord):
    # create only 7x7 subfield from 17x17
    #
    # Expand 17x17 matrix to 19x19
    field_19x19 = np.full(shape=(19, 19), fill_value=-1)
    # Fill 19x19 matrix with values from field
    exp_val = 2
    field_19x19[exp_val:exp_val + field.shape[0], exp_val:exp_val + field.shape[1]] = field

    # get 7x7 from 19x19 matrix based on own position
    x = self_coord[0] + exp_val
    y = self_coord[0] + exp_val

    field_7x7 = field_19x19[x - 3:x + 3 + 1, y - 3:y + 3 + 1]
    return field_7x7