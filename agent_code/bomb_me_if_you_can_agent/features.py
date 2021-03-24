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
                up_steps = up_steps + 1
        if down:
            if field[own_pos[1] + i, own_pos[0]] == -1 or field[own_pos[1] + i, own_pos[0]] == 1:
                down = False
            elif field[own_pos[1] + i, own_pos[0]] == 0:
                down_steps = down_steps + 1
        if left:
            if field[own_pos[1], own_pos[0] - i] == -1 or field[own_pos[1], own_pos[0] - i] == 1:
                left = False
            elif field[own_pos[1], own_pos[0] - i] == 0:
                left_steps = left_steps + 1
        if right:
            if field[own_pos[1], own_pos[0] + i] == -1 or field[own_pos[1], own_pos[0] + i] == 1:
                right = False
            elif field[own_pos[1], own_pos[0] + i] == 0:
                right_steps = right_steps + 1

    actions = np.asarray([up_steps, down_steps, left_steps, right_steps])
    bomb_possible = int(game_state.get('self')[2] == True)

    # find_3 neighbours_values delivers an array in the form [up,down,left,right]
    free_dir = find_3neighbor_values(self_coord=own_pos, field_ch=field)
    free_dir = [[up_steps, free_dir[0], down_steps], [free_dir[2], bomb_possible, free_dir[3]],
                [left_steps, free_dir[1], right_steps]]

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

    # calculate explosion of bombs
    field = game_state.get('field').T
    bombs = game_state.get('bombs')
    bomb_field = np.zeros_like(field)

    # calculate field with all bomb and expected explosions
    up, down, left, right = True, True, True, True
    if len(bombs) > 0:
        for bomb in bombs:
            bomb_timer = bomb[1] + 3
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

    # feature enemies
    # get orientation of nearest_coin
    down, up, left, left_up, left_down, right, right_up, right_down = 0, 0, 0, 0, 0, 0, 0, 0
    enemies = game_state.get('others')
    man_dis = []
    if len(enemies) > 0:
        for enemy in enemies:
            field[enemy[3][1], enemy[3][0]] = 1
            # determine manhattan distance to each coin on field
            man_dis.append(distance.cityblock([own_pos[1], own_pos[0]], [enemy[3][1], enemy[3][0]]))

    nearest_enemy_dist = 0

    if len(man_dis) > 0:
        # select coin with shortest distance
        nearest_enemy = np.argmin(man_dis)
        nearest_enemy_dist = man_dis[nearest_enemy]
        nearest_enemy_pos = enemies[nearest_enemy][3]

        # Determine Orientation of coin
        if own_pos[1] == nearest_enemy_pos[1]:
            if own_pos[0] >= nearest_enemy_pos[0]:
                left = 1
            else:
                right = 1
        elif own_pos[0] == nearest_enemy_pos[0]:
            if own_pos[1] >= nearest_enemy_pos[1]:
                up = 1
            else:
                down = 1
        elif own_pos[0] < nearest_enemy_pos[0] and own_pos[1] < nearest_enemy_pos[1]:
            right_down = 1
        elif own_pos[0] < nearest_enemy_pos[0] and own_pos[1] > nearest_enemy_pos[1]:
            right_up = 1
        elif own_pos[0] > nearest_enemy_pos[0] and own_pos[1] > nearest_enemy_pos[1]:
            left_up = 1
        elif own_pos[0] > nearest_enemy_pos[0] and own_pos[1] < nearest_enemy_pos[1]:
            left_down = 1

    enemies = np.asarray([[left_up, up, right_up], [left, 0, right], [left_down, down, right_down]])
    mask = enemies == 1
    enemies[mask] = nearest_enemy_dist

    # transform np-arrays to tensors
    free_dir = torch.tensor(free_dir).double()
    crates = torch.tensor(crates).double()  # crates instead of nearest crates to give model more information
    bombs = torch.tensor(bombs).double()
    coins = torch.tensor(coins).double()
    actions = torch.tensor(actions).double()
    enemies = torch.tensor(enemies).double()

    # stacked_channels = torch.stack((self_ch_ten, other0_ch_ten, bombs_ch_ten, coins_ch_ten, explosion_map_ch_ten), 0)
    stacked_channels = torch.stack((free_dir, coins, bombs, crates, enemies), 0)
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

    # field = get_7x7_submatrix(field,self_coord)
    # others = get_7x7_submatrix(other_field,self_coord)
    # bombs = get_7x7_submatrix(bomb_field,self_coord)
    # coins = get_7x7_submatrix(coins,self_coord)

    field = torch.tensor(field).double()
    # self = torch.tensor(self).double()
    others = torch.tensor(other_field).double()
    bombs = torch.tensor(bomb_field).double()
    coins = torch.tensor(coins).double()

    field = torch.tensor(field).double()
    tensor_ch = torch.stack(
        (field, others, bombs, coins), 0)

    tensor_ch = tensor_ch.unsqueeze(0)

    return tensor_ch


def get_7x7_submatrix(field, self_coord):
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

    if (field_ch[up_list_coord[0][1], up_list_coord[0][0]] != 0):
        up_list = np.ones(len(up_list_coord))
    else:
        if (field_ch[up_list_coord[1][1], up_list_coord[1][0]] != 0):
            up_list[1] = 1
            up_list[4] = 1
            up_list[5] = 1
            up_list[6] = 1
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

    if (field_ch[down_list_coord[0][1], down_list_coord[0][0]] != 0):
        down_list = np.ones(len(down_list_coord))
    else:
        if (field_ch[down_list_coord[1][1], down_list_coord[1][0]] != 0):
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

    if (field_ch[left_list_coord[0][1], left_list_coord[0][0]] != 0):
        left_list = np.ones(len(left_list_coord))
    else:
        if (field_ch[left_list_coord[1][1], left_list_coord[1][0]] != 0):
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

    if (field_ch[right_list_coord[0][1], right_list_coord[0][0]] != 0):
        right_list = np.ones(len(right_list_coord))
    else:
        if (field_ch[right_list_coord[1][1], right_list_coord[1][0]] != 0):
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

        duplicate_tuple = up_list_coord[duplicates_up_list[i]]
        ind_to_check = [i for i, tupl in enumerate(up_list_coord) if tupl == duplicate_tuple]
        if (up_list[ind_to_check[0]] == 1 and up_list[ind_to_check[1]] == 1):
            up_list[ind_to_check[1]] = 1
        if (up_list[ind_to_check[0]] == 5 and up_list[ind_to_check[1]] == 1):
            up_list[ind_to_check[1]] = 1
        if (up_list[ind_to_check[1]] == 5 and up_list[ind_to_check[0]] == 1):
            up_list[ind_to_check[0]] = 1
        if (up_list[ind_to_check[1]] == 5 and up_list[ind_to_check[0]] == 5):
            up_list[ind_to_check[1]] = 1

    for j in range(len(duplicates_down_list)):

        duplicate_tuple = down_list_coord[duplicates_down_list[j]]
        ind_to_check = [i for i, tupl in enumerate(down_list_coord) if tupl == duplicate_tuple]

        if (down_list[ind_to_check[0]] == 1 and down_list[ind_to_check[1]] == 1):
            down_list[ind_to_check[1]] = 1
        if (down_list[ind_to_check[0]] == 5 and down_list[ind_to_check[1]] == 1):
            down_list[ind_to_check[1]] = 1
        if (down_list[ind_to_check[1]] == 5 and down_list[ind_to_check[0]] == 1):
            down_list[ind_to_check[0]] = 1
        if (down_list[ind_to_check[1]] == 5 and down_list[ind_to_check[0]] == 5):
            down_list[ind_to_check[1]] = 1

    for k in range(len(duplicates_left_list)):

        duplicate_tuple = left_list_coord[duplicates_left_list[k]]
        ind_to_check = [i for i, tupl in enumerate(left_list_coord) if tupl == duplicate_tuple]

        if (left_list[ind_to_check[0]] == 1 and left_list[ind_to_check[1]] == 1):
            left_list[ind_to_check[1]] = 1
        if (left_list[ind_to_check[0]] == 5 and left_list[ind_to_check[1]] == 1):
            left_list[ind_to_check[1]] = 1
        if (left_list[ind_to_check[1]] == 5 and left_list[ind_to_check[0]] == 1):
            left_list[ind_to_check[0]] = 1
        if (left_list[ind_to_check[1]] == 5 and left_list[ind_to_check[0]] == 5):
            left_list[ind_to_check[1]] = 1

    for l in range(len(duplicates_right_list)):

        duplicate_tuple = right_list_coord[duplicates_right_list[l]]
        ind_to_check = [i for i, tupl in enumerate(right_list_coord) if tupl == duplicate_tuple]

        if (right_list[ind_to_check[0]] == 1 and right_list[ind_to_check[1]] == 1):
            right_list[ind_to_check[1]] = 1
        if (right_list[ind_to_check[0]] == 5 and right_list[ind_to_check[1]] == 1):
            right_list[ind_to_check[1]] = 1
        if (right_list[ind_to_check[1]] == 5 and right_list[ind_to_check[0]] == 1):
            right_list[ind_to_check[0]] = 1
        if (right_list[ind_to_check[1]] == 5 and right_list[ind_to_check[0]] == 5):
            right_list[ind_to_check[1]] = 1

    list_of_lists_coord = [up_list_coord, down_list_coord, left_list_coord, right_list_coord]
    list_of_lists_val = [up_list, down_list, left_list, right_list]

    final_res = []

    for i in range(len(list_of_lists_coord)):
        counter = 0
        for j in range(len(list_of_lists_coord[i])):
            if ((list_of_lists_coord[i][j][1] >= 1 and list_of_lists_coord[i][j][1] <= 15 and list_of_lists_coord[i][j][
                0] >= 1 and list_of_lists_coord[i][j][0] <= 15) and (
                    field_ch[list_of_lists_coord[i][j][1], list_of_lists_coord[i][j][0]] == 0) and (
                    list_of_lists_val[i][j] == 5)):
                counter += 1
        final_res.append(counter)

    return final_res
