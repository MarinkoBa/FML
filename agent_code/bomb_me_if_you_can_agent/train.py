import pickle
import random
from collections import namedtuple, deque
from typing import List

import torch
import torch.nn as nn

import events as e
from .callbacks import state_to_features
from agent_code.bomb_me_if_you_can_agent import constants

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
    if ...:
        events.append(PLACEHOLDER_EVENT)


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

    action = last_action
    if(last_action != None):
        action = self.actions.index(last_action)

    transition = Transition(state_to_features(last_game_state), action, None, reward_from_events(self, events))
    if transition[0] != None:
        self.transitions.append(transition)

    constants.ROUNDS_NR = constants.ROUNDS_NR + 1

    if(constants.ROUNDS_NR%3 == 0):
        self.target_model.load_state_dict(self.trainings_model.state_dict())

    sample_batch_and_train(self)


    print()
    print(str(constants.EPS))
    print()


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.trainings_model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        #e.COIN_COLLECTED: 10,
        #e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -15,
        #e.WAITED: -5,
        e.WAITED: -5,
        e.MOVED_UP: 5,
        e.MOVED_DOWN: 5,
        e.MOVED_LEFT: 5,
        e.MOVED_RIGHT: 5,
        #e.COIN_FOUND: 0.5,
        #e.CRATE_DESTROYED: 0.5,
        #e.GOT_KILLED: -50,
        #e.KILLED_SELF: 5,
        #e.SURVIVED_ROUND: 10
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def sample_batch_and_train(self):

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
        q_val_taken_actions = self.trainings_model(game_state.to(device='cuda:0')).gather(1, action.to(
            device='cuda:0').unsqueeze(1))

        q_val_next_state = torch.max(self.target_model(next_game_state), dim=1).values.to(device='cuda')
        q_val_next_state_full = torch.zeros(game_state.shape[0]).double().to(device='cuda')
        q_val_next_state_full[next_game_state_filled_ind] = q_val_next_state

        final_state_action_values = (q_val_next_state_full * constants.GAMMA) + reward.to(device='cuda')

        loss = self.criterion(q_val_taken_actions, final_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




