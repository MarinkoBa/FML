import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import torch
import torch.nn as nn
import settings as s

import events as e
from .callbacks import state_to_features
from agent_code.bomb_me_if_you_can_agent import constants
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

PLACEHOLDER_EVENT = 'PLACEHOLDER'


def setup_training(self):
    """
    Initialise self for training purpose.a
    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=constants.TRANSITION_HISTORY_SIZE)
    self.penultimate_position = (0,0)

    self.round_events = []
    self.rewards_list = []
    self.reward_mean = []
    self.rewards_list_game = []
    self.reward_mean_game = []
    self.penultimate_position = (0, 0)

    # setup plot

    # plt.title('Q-Net Training')
    # plt.xlabel('Episode')
    # plt.ylabel('Rewards')
    # plt.ylim([-constants.SIZE_Y_AXIS, constants.SIZE_Y_AXIS])
    # plt.ion()

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.title('Rewards training')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.ylim([-constants.SIZE_Y_AXIS, constants.SIZE_Y_AXIS])
    plt.ion()

    plt.subplot(122)
    plt.title('Rewards game')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.ylim([0, 30])
    plt.ion()

    plt.suptitle('Q-Net Training')

    # Pre Training:
    # load_training_data(self)
    # for i in range(100):
    #     print('Pre Training Round: '+str(i))
    #     train_neural_network(self)


def game_events_occurred(self, old_game_state: dict, self_action: str, next_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param next_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {next_game_state["step"]}')


    # Idea: Add your own events to hand out rewards
    # CHECK PLACEMENT OF THE BOMB
    if old_game_state is not None:

        old_field = old_game_state.get('field').T
        bombs = next_game_state.get('bombs')
        for i in range(len(bombs)):
            if bombs[i][1] == 3:
                if (old_field[bombs[i][0][1] + 1, bombs[i][0][0]] == 1 or old_field[
                    bombs[i][0][1] - 1, bombs[i][0][0]] == 1 or
                    old_field[bombs[i][0][1], bombs[i][0][0] + 1] == 1 or old_field[
                        bombs[i][0][1], bombs[i][0][0] - 1] == 1) and (
                        e.BOMB_DROPPED in events) and old_game_state.get('self')[3] == bombs[i][0]:
                    events.append(e.BOMB_PLACED_AT_CRATE)


    if next_game_state.get('step') >= 2:
        if next_game_state.get('self')[3] == self.penultimate_position and e.BOMB_DROPPED not in events and e.WAITED not in events:
            events.append(e.RETURN_TO_PREVIOUS_POS)
        self.penultimate_position = old_game_state.get('self')[3]

    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(e.SURVIVED_BOMB)
    if e.BOMB_DROPPED in events and old_game_state.get('step') == 1:
        events.append(e.PLACED_BOMB_FIRST_STEP)

    for event in events:
        self.round_events.append(event)
    # state_to_features is defined in callbacks.py
    action = self_action
    if (self_action != None):
        action = self.actions.index(self_action)

    transition = Transition(state_to_features(old_game_state), action, state_to_features(next_game_state),
                            reward_from_events(self, events))
    if transition[0] != None:
        self.transitions.append(transition)

    train_neural_network(self)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # Determine Rewards from whole episode
    for event in events:
        self.round_events.append(event)

    self.rewards_list.append(reward_from_events(self, self.round_events))

    sum_reward_round = 0
    for reward in self.round_events:
        if (reward == e.COIN_COLLECTED):
            sum_reward_round += 1
        elif (reward == e.KILLED_OPPONENT):
            sum_reward_round += 5
    self.rewards_list_game.append(sum_reward_round)

    self.round_events = []

    if last_game_state.get('round') % constants.PLOT_MEAN_OVER_ROUNDS == 0:
        self.reward_mean.append(np.mean(self.rewards_list[-constants.PLOT_MEAN_OVER_ROUNDS:]))
        self.reward_mean_game.append(np.mean(self.rewards_list_game[-constants.PLOT_MEAN_OVER_ROUNDS:]))

    if last_game_state.get('round') % constants.EPISODES_TO_PLOT == 0:
        plt.subplot(121)
        plt.plot(self.rewards_list)
        plt.plot(np.arange(0, len(self.reward_mean)) * constants.PLOT_MEAN_OVER_ROUNDS, self.reward_mean)

        plt.subplot(122)
        plt.plot(self.rewards_list_game)
        plt.plot(np.arange(0, len(self.reward_mean_game)) * constants.PLOT_MEAN_OVER_ROUNDS, self.reward_mean_game)
        plt.savefig(constants.NAME_OF_FILES + '_plot.png')

    action = last_action
    if (last_action != None):
        action = self.actions.index(last_action)

    transition = Transition(state_to_features(last_game_state), action, None, reward_from_events(self, events))

    if transition[0] != None:
        self.transitions.append(transition)

    if constants.EPS >= constants.EPS_END:
        constants.EPS = constants.EPS - (constants.EPS / constants.EPS_DECAY)

    if last_game_state.get('round') % constants.ROUNDS_MODEL_UPDATE == 0:
        self.target_model.load_state_dict(self.training_model.state_dict())
        print()
        print('UPDATE TARGET MODEL')
        print()

    # Increase Steps per Round after particular Episodes
    if constants.ROUNDS_NR % constants.INCREASE_STEPS_AFTER_EPISODES == 0 and s.MAX_STEPS < 400:
        s.MAX_STEPS = s.MAX_STEPS + constants.INCREASE_STEP_VALUE
    constants.ROUNDS_NR = constants.ROUNDS_NR +1

    print()
    print('Epsilon: ' + str(constants.EPS))
    print()


    # Store the model
    with open(constants.NAME_OF_FILES + ".pt", "wb") as file:
        pickle.dump(self.training_model.cpu(), file)
    self.training_model.cuda()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 0.8,
        e.KILLED_OPPONENT: 0.95,
        e.INVALID_ACTION: -0.9,  # macht es sinn invalide aktionen zu bestrafen?
        e.COIN_FOUND: 0.01,
        e.CRATE_DESTROYED: 0.3,
        e.GOT_KILLED: -0.90,
        e.KILLED_SELF: -1,
        e.SURVIVED_ROUND: 0.8,
        e.OPPONENT_ELIMINATED: 0.05,  # nicht durch unsern agent direkt gekillt
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.SURVIVED_BOMB: 0.1,
        e.PLACED_BOMB_FIRST_STEP: -0.7,  # Bomb in first step, is at all time bad
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.WAITED: -0.4,
        e.BOMB_PLACED_AT_CRATE: 0.3,
        e.RETURN_TO_PREVIOUS_POS: -0.4
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def sample_batch(self):
    return random.sample(self.transitions, constants.BATCH_SIZE)

def load_training_data(self):
    with open("training_data.pt", "rb") as file:
        transitions = pickle.load(file)

    for transition in transitions:
        trans = Transition(state_to_features(transition[0]), transition[1], state_to_features(transition[2]), reward_from_events(self, transition[3]))
        self.transitions.append(trans)

def train_neural_network(self):
    # TODO check if Batch filled, train
    if len(self.transitions) >= constants.BATCH_SIZE and self.train:
        batch = random.sample(self.transitions, constants.BATCH_SIZE)
        transitions = Transition(*zip(*batch))

        next_game_state_filled_ind = [i for i, val in enumerate(transitions.next_state) if val != None]
        next_game_state_filled_val = [val for i, val in enumerate(transitions.next_state) if val != None]

        game_state = torch.cat(transitions.state)
        action = torch.tensor(transitions.action)
        reward = torch.tensor(transitions.reward)
        next_game_state = torch.cat(next_game_state_filled_val)

        # TODO modify input -> tensor 128x7x17x17
        q_val_taken_actions = self.training_model(game_state.to(device='cuda:0')).gather(1, action.to(
            device='cuda:0').unsqueeze(1))

        q_val_next_state = torch.max(self.target_model(next_game_state), dim=1).values.to(device='cuda')
        q_val_next_state_full = torch.zeros(game_state.shape[0]).double().to(device='cuda')
        q_val_next_state_full[next_game_state_filled_ind] = q_val_next_state

        final_state_action_values = (q_val_next_state_full * constants.GAMMA) + reward.to(device='cuda')

        loss = self.criterion(q_val_taken_actions, final_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.training_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()