import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import settings as s

import matplotlib.pyplot as plt
import torch

import events as e
from .callbacks import state_to_features
from . import constants

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


PLACEHOLDER_EVENT = 'PLACEHOLDER'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=constants.TRANSITION_HISTORY_SIZE)


    self.round_events = []
    self.rewards_list = []
    self.reward_mean = []
    self.rewards_list_game = []
    self.steps_list_game = []
    self.reward_mean_game = []
    self.penultimate_position = (0,0)

    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.title('Steps per round')
    plt.xlabel('Episode')
    plt.ylabel('Step')
    plt.ylim([0, 410])
    plt.ion()
    ax = plt.gca()

    plt.subplot(132)
    plt.title('Rewards game')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.ylim([0, 30])
    plt.ion()
    ax = plt.gca()

    plt.subplot(133)
    plt.title('Score after 1000 round')
    plt.ylabel('Score')
    plt.ylim([0, 24500])
    plt.ion()
    ax = plt.gca()

    plt.suptitle('Q-Net Test')



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

    if(old_game_state!=None and next_game_state!=None):
        # SURVIVED OWN BOMB
        if(old_game_state['self'][2]==False and next_game_state['self'][2]==True):
            events.append(e.SURVIVED_BOMB)


        # CHECK PLACEMENT OF THE BOMB -> Bad placed vs good placed reward
        old_field = old_game_state.get('field').T
        bombs = next_game_state.get('bombs')
        for i in range(len(bombs)):
            if bombs[i][1] == 3:
                if (old_field[bombs[i][0][1] + 1, bombs[i][0][0]]==1 or old_field[bombs[i][0][1] - 1, bombs[i][0][0]]==1 or
                    old_field[bombs[i][0][1], bombs[i][0][0] + 1]==1 or old_field[bombs[i][0][1], bombs[i][0][0] - 1]==1) and (
                        e.BOMB_DROPPED in events) and old_game_state.get('self')[3]==bombs[i][0]:
                    events.append(e.BOMB_PLACED_AT_CRATE)

                elif (old_field[bombs[i][0][1] + 1, bombs[i][0][0]]!=1 and old_field[bombs[i][0][1] - 1, bombs[i][0][0]]!=1 and
                    old_field[bombs[i][0][1], bombs[i][0][0] + 1]!=1 and old_field[bombs[i][0][1], bombs[i][0][0] - 1]!=1) and (
                        e.BOMB_DROPPED in events) and old_game_state.get('self')[3]==bombs[i][0]:
                    events.append(e.BOMB_PLACED_BAD)

        # check if old position reached
        if next_game_state.get('step') >= 2:
            if next_game_state.get('self')[3] == self.penultimate_position and e.BOMB_DROPPED not in events and e.WAITED not in events:
                events.append(e.RETURN_TO_PREVIOUS_POS)
            self.penultimate_position = old_game_state.get('self')[3]

        # CHECK IF MOVED TOWARDS CRATE
        self_coord = old_game_state.get('self')[3]
        old_fieldup = old_game_state.get('field').T[self_coord[1] - 1, self_coord[0]]
        old_fielddown = old_game_state.get('field').T[self_coord[1] + 1, self_coord[0]]
        old_fieldright = old_game_state.get('field').T[self_coord[1], self_coord[0] + 1]
        old_fieldleft = old_game_state.get('field').T[self_coord[1], self_coord[0] - 1]

        new_fieldup = 0
        new_fieldup2 = 0
        new_fieldup3 = 0
        new_fieldup4 = 0
        new_fielddown = 0
        new_fielddown2 = 0
        new_fielddown3 = 0
        new_fielddown4 = 0
        new_fieldright = 0
        new_fieldright2 = 0
        new_fieldright3 = 0
        new_fieldright4 = 0
        new_fieldleft = 0
        new_fieldleft2 = 0
        new_fieldleft3 = 0
        new_fieldleft4 = 0
        if(e.MOVED_LEFT in events):
            new_fieldup = next_game_state.get('field').T[self_coord[1] - 1, self_coord[0]-1]
            new_fielddown = next_game_state.get('field').T[self_coord[1] + 1, self_coord[0]-1]
            new_fieldright = next_game_state.get('field').T[self_coord[1], self_coord[0] + 1 - 1]
            new_fieldleft = next_game_state.get('field').T[self_coord[1], self_coord[0] - 1 - 1]

            if(self_coord[0] - 1) > 5:
                new_fieldleft2 = next_game_state.get('field').T[self_coord[1], self_coord[0] - 1 - 1 - 1]
                new_fieldleft3 = next_game_state.get('field').T[self_coord[1], self_coord[0] - 1 - 1 - 1 -1]
                new_fieldleft4 = next_game_state.get('field').T[self_coord[1], self_coord[0] - 1 - 1 - 1 -1 -1]

        if(e.MOVED_RIGHT in events):
            new_fieldup = next_game_state.get('field').T[self_coord[1] - 1, self_coord[0]+1]
            new_fielddown = next_game_state.get('field').T[self_coord[1] + 1, self_coord[0]+1]
            new_fieldright = next_game_state.get('field').T[self_coord[1], self_coord[0] + 1 + 1]
            new_fieldleft = next_game_state.get('field').T[self_coord[1], self_coord[0] - 1 + 1]

            if(self_coord[0] + 1) < 10:
                new_fieldright2 = next_game_state.get('field').T[self_coord[1], self_coord[0] + 1 + 1 +1]
                new_fieldright3 = next_game_state.get('field').T[self_coord[1], self_coord[0] + 1 + 1 +1 +1]
                new_fieldright4 = next_game_state.get('field').T[self_coord[1], self_coord[0] + 1 + 1 +1 +1 +1]


        if(e.MOVED_UP in events):
            new_fieldup = next_game_state.get('field').T[self_coord[1] - 1 - 1 , self_coord[0]]
            new_fielddown = next_game_state.get('field').T[self_coord[1] + 1 -1, self_coord[0]]
            new_fieldright = next_game_state.get('field').T[self_coord[1] -1, self_coord[0] + 1]
            new_fieldleft = next_game_state.get('field').T[self_coord[1] -1, self_coord[0] - 1]

            if(self_coord[1] - 1) > 5:
                new_fieldup2 = next_game_state.get('field').T[self_coord[1] - 1 - 1 -1, self_coord[0]]
                new_fieldup3 = next_game_state.get('field').T[self_coord[1] - 1 - 1 -1 -1, self_coord[0]]
                new_fieldup4 = next_game_state.get('field').T[self_coord[1] - 1 - 1 -1 -1 -1, self_coord[0]]

        if(e.MOVED_DOWN in events):
            new_fieldup = next_game_state.get('field').T[self_coord[1] - 1 + 1 , self_coord[0]]
            new_fielddown = next_game_state.get('field').T[self_coord[1] + 1 +1, self_coord[0]]
            new_fieldright = next_game_state.get('field').T[self_coord[1] +1, self_coord[0] + 1]
            new_fieldleft = next_game_state.get('field').T[self_coord[1] +1, self_coord[0] - 1]

            if(self_coord[1] + 1) < 10:
                new_fielddown2 = next_game_state.get('field').T[self_coord[1] + 1 +1 +1, self_coord[0]]
                new_fielddown3 = next_game_state.get('field').T[self_coord[1] + 1 +1 +1 +1, self_coord[0]]
                new_fielddown4 = next_game_state.get('field').T[self_coord[1] + 1 +1 +1 +1 +1, self_coord[0]]

        if(e.MOVED_DOWN in events) or (e.MOVED_UP in events) or (e.MOVED_LEFT in events) or (e.MOVED_RIGHT in events):
            if (old_fieldup != 1 and old_fielddown != 1 and old_fieldleft != 1 and old_fieldright != 1) and \
                    (new_fieldup == 1 or new_fieldup2 == 1 or new_fieldup3 == 1  or new_fieldup4 == 1
                     or new_fielddown == 1 or new_fielddown2 == 1 or new_fielddown3 == 1 or new_fielddown4 == 1
                     or new_fieldleft == 1 or new_fieldleft2 == 1 or new_fieldleft3 == 1 or new_fieldleft4 == 1
                    or new_fieldright == 1 or new_fieldright2 == 1 or new_fieldright3 == 1 or new_fieldright4 == 1):
                events.append(e.MOVED_TOWARDS_CRATE)

            if (new_fieldup != 1 and new_fieldup2 != 1 and new_fieldup3 != 1 and new_fieldup4 != 1
                     and new_fielddown != 1 and new_fielddown2 != 1 and new_fielddown3 != 1 and new_fielddown4 != 1
                     and new_fieldleft != 1 and new_fieldleft2 != 1 and new_fieldleft3 != 1 and new_fieldleft4 != 1
                    and new_fieldright != 1 and new_fieldright2 != 1 and new_fieldright3 != 1 and new_fieldright3 != 4):

                bomb_was_there = False
                for i in range(len(bombs)):
                    if(bombs[i][0] == self_coord):
                        bomb_was_there = True
                if not bomb_was_there:
                    events.append(e.MOVED_AWAY_FROM_CRATE)
        ##########

    for event in events:
        self.round_events.append(event)

    if constants.EPS >= constants.EPS_END:
        constants.EPS = constants.EPS - (constants.EPS / 10000)

    # state_to_features is defined in callbacks.py
    action = self_action
    if(self_action != None):
        action = self.actions.index(self_action)

    transition = Transition(state_to_features(old_game_state), action, state_to_features(next_game_state), reward_from_events(self, events))
    if transition[0] != None:
        self.transitions.append(transition)


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

    for event in events:
        self.round_events.append(event)

    self.rewards_list.append(reward_from_events(self, self.round_events))


    sum_reward_round = 0
    for reward in self.round_events:
        if(reward==e.COIN_COLLECTED):
            sum_reward_round+=1
        elif(reward==e.KILLED_OPPONENT):
            sum_reward_round+=5
    self.rewards_list_game.append(sum_reward_round)

    self.round_events = []

    self.steps_list_game.append(last_game_state['step'])

    if last_game_state.get('round') % constants.PLOT_MEAN_OVER_ROUNDS == 0:
        self.reward_mean.append(np.mean(self.rewards_list[-constants.PLOT_MEAN_OVER_ROUNDS:]))
        self.reward_mean_game.append(np.mean(self.rewards_list_game[-constants.PLOT_MEAN_OVER_ROUNDS:]))

    if last_game_state.get('round') % constants.EPISODES_TO_PLOT == 0:
        plt.subplot(131)
        plt.plot(np.arange(1, len(self.steps_list_game)+1), self.steps_list_game)
        plt.xticks(np.arange(1, len(self.steps_list_game)+1, 1))

        plt.subplot(132)
        plt.plot(np.arange(1, len(self.rewards_list_game)+1), self.rewards_list_game)
        plt.xticks(np.arange(1, len(self.rewards_list_game)+1, 1))

        plt.subplot(133)
        plt.xlim(0.5, 1.5)
        plt.bar([1], [np.sum(self.rewards_list_game)], width=0.7)
        plt.xticks([])
        plt.xlabel('1000 rounds')

        plt.savefig('19_crate_3peaceful_agents_EVAL.png')

    action = last_action
    if(last_action != None):
        action = self.actions.index(last_action)

    transition = Transition(state_to_features(last_game_state), action, None, reward_from_events(self, events))
    if transition[0] != None:
        self.transitions.append(transition)

    constants.ROUNDS_NR = constants.ROUNDS_NR + 1

    if(constants.ROUNDS_NR%50 == 0):
        self.target_model.load_state_dict(self.trainings_model.state_dict())

    if constants.ROUNDS_NR%5000 == 0:
        s.MAX_STEPS = s.MAX_STEPS + 20
        constants.EPS = 0.6
        if s.MAX_STEPS >= 400:
            s.MAX_STEPS=400

    sample_batch_and_train(self)

    print()
    print(str(constants.EPS))
    print()


    # Store the model
    with open("B_M_I_Y_C_agent.pt", "wb") as file:
        pickle.dump(self.trainings_model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

#LAST VERSION GAME REWARDS CRATE DESTROYER AND AGAINST AGENTS!
    game_rewards = {
        e.WAITED: -25,
        e.MOVED_UP: -15,
        e.MOVED_DOWN: -15,
        e.MOVED_LEFT: -15,
        e.MOVED_RIGHT: -15,

        e.INVALID_ACTION: -50,

        e.COIN_COLLECTED: 350,
        e.COIN_FOUND: 50,

        e.KILLED_OPPONENT: 500,
        e.KILLED_SELF: -300,
        e.SURVIVED_ROUND: 50,
        e.GOT_KILLED: -100,
        e.SURVIVED_BOMB: 100,

        e.CRATE_DESTROYED: 35,

        e.BOMB_DROPPED: 0,
        e.BOMB_PLACED_AT_CRATE: 125,
        e.BOMB_PLACED_BAD: -85,

        e.MOVED_TOWARDS_CRATE: 85,
        e.RETURN_TO_PREVIOUS_POS: -35,
        e.MOVED_AWAY_FROM_CRATE: -30
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def sample_batch_and_train(self):
    '''
       Method updates the training_model, based on the training data which was collected by the agent during playing the game via Exploration/Explotation.
       Parameters of the training such as Batch size, learning rate and so on can be modified in the constants.py
    '''

    if len(self.transitions) >= constants.BATCH_SIZE and self.train:
        batch = random.sample(self.transitions, constants.BATCH_SIZE)
        transitions = Transition(*zip(*batch))

        next_game_state_filled_ind = [i for i, val in enumerate(transitions.next_state) if val != None]
        next_game_state_filled_val = [val for i, val in enumerate(transitions.next_state) if val != None]

        game_state = torch.cat(transitions.state)
        action = torch.tensor(transitions.action)
        reward = torch.tensor(transitions.reward)
        next_game_state = torch.cat(next_game_state_filled_val)

        q_val_taken_actions = self.trainings_model(game_state.to(device='cuda:0')).gather(1, action.to(
            device='cuda:0').unsqueeze(1))

        q_val_next_state = torch.max(self.target_model(next_game_state), dim=1).values.to(device='cuda')
        q_val_next_state_full = torch.zeros(game_state.shape[0]).double().to(device='cuda')
        q_val_next_state_full[next_game_state_filled_ind] = q_val_next_state

        final_state_action_values = (q_val_next_state_full * constants.GAMMA) + reward.to(device='cuda')

        loss = self.criterion(q_val_taken_actions, final_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K
    and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val / 1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val / 1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    new_tick_format = str(new_tick_format)

    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]

    return new_tick_format