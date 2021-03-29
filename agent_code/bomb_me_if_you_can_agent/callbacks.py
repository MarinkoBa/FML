import os
import pickle
import random
from agent_code.bomb_me_if_you_can_agent import features
import numpy as np
import torch
from torch import nn
from agent_code.bomb_me_if_you_can_agent.q_lin_network import Q_Net
import time
from agent_code.bomb_me_if_you_can_agent import constants

from collections import deque
from random import shuffle


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
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    if self.train or not os.path.isfile("5_layer_multi_enemy_destroyer_train_rule_based_decaybigger.pt"):
        self.logger.info("Setting up model from scratch.")

        # if you want to train on a existing model
        # with open("5_layer_multi_enemy_destroyer_train_rule_based_decaybigger.pt", "rb") as file:
        #     self.target_model = pickle.load(file)
        # self.target_model.cuda()
        # self.training_model = self.target_model

        self.training_model = Q_Net()
        self.training_model.cuda(0)
        self.training_model.train()
        self.training_model.double()

        self.target_model = Q_Net()
        self.target_model.cuda()
        self.target_model.double()
        self.target_model.load_state_dict(self.training_model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(params=self.training_model.parameters(), lr=constants.LEARNING_RATE)
        self.criterion = nn.SmoothL1Loss()
        self.criterion.cuda(0)

        self.actions = constants.ACTIONS
    else:
        self.logger.info("Loading model from saved state.")
        with open("5_layer_multi_enemy_destroyer_train_rule_based_decaybigger.pt", "rb") as file:
            self.target_model = pickle.load(file)
        self.target_model.cuda()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # Exploration vs exploitation
    random_prob = constants.EPS
    if self.train and random.random() < random_prob:
        # decide which way of exploration you want to use
        return valid_actions(game_state)
        #return rule_based_exploration(self, game_state)

    game_state_features = state_to_features(game_state)
    prediction = self.target_model(game_state_features)
    action_pos = torch.argmax(prediction)

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

    stacked_channels = features.features_3x3_space(game_state)
    stacked_channels = stacked_channels.unsqueeze(0)
    #stacked_channels = features.feature_field(game_state)

    # and return them as a vector
    return stacked_channels


def valid_actions(game_state):
    '''
    This method allows only to choose random valid moves.
    Args:
        game_state: Includes all information about the current game state/ all states on the game board

    Returns:
        String, One valid action from the ACTIONS List ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    '''
    field = game_state.get('field')
    own_pos = game_state.get('self')[3]

    # check if neighboured fields are valid
    down = field[own_pos[1] + 1, own_pos[0]] == 0
    up = field[own_pos[1] - 1, own_pos[0]] == 0
    left = field[own_pos[1], own_pos[0] - 1] == 0
    right = field[own_pos[1], own_pos[0] + 1] == 0

    count = np.sum([up, right, down, left])
    if count == 0:
        probabilities = [0, 0, 0, 0, 0.5, 0.5]
    else:
        probabilities = [0.8 * up / count, 0.8 * right / count, 0.8 * down / count, 0.8 * left / count, 0.1, 0.1]

    return np.random.choice(constants.ACTIONS, p=probabilities)


def rule_based_exploration(self, game_state):
    '''
    Method, execute the Exploration based on the rule_based agent.
    Args:
        self:
        game_state: Includes all information about the current game state/ all states on the game board

    Returns:
        String, a valid action, which the rule_based_agent would execute in the actual game state.


    '''
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]