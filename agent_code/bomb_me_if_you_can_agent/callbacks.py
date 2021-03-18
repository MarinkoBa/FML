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

        self.training_model = Q_Net()
        self.training_model.cuda(0)
        self.training_model.train()
        self.training_model.double()

        self.optimizer = torch.optim.Adam(params=self.training_model.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss()
        self.criterion.cuda(0)

        self.target_model = Q_Net()
        self.target_model.cuda()
        self.target_model.double()
        self.target_model.load_state_dict(self.training_model.state_dict())
        self.target_model.eval()

        self.actions = constants.ACTIONS
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.training_model = pickle.load(file)


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

    stacked_channels = features.features_3x3_space(game_state)
    stacked_channels = stacked_channels.unsqueeze(0)
    # stacked_channels = stacked_channels.unsqueeze(0)

    # and return them as a vector
    return stacked_channels


def valid_actions(game_state):
    field = game_state.get('field')
    own_pos = game_state.get('self')[3]

    # check if neighboured fields are valid
    down = field[own_pos[1] + 1, own_pos[0]] == 0
    up = field[own_pos[1] - 1, own_pos[0]] == 0
    left = field[own_pos[1], own_pos[0] - 1] == 0
    right = field[own_pos[1], own_pos[0] + 1] == 0

    count = np.sum([up, right, down, left])

    return [0.8 * up / count, 0.8 * right / count, 0.8 * down / count, 0.8 * left / count, 0.1, 0.1]
