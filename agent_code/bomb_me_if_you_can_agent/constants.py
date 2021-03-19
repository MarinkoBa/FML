AMOUNT_OF_TRAINING_EPISODES = 100000
NAME_OF_FILES = 'my-saved-model'

# Hyper parameters
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

BATCH_SIZE = 128
GAMMA = 0.9
EPS = 1
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10


ROUNDS_MODEL_UPDATE = 5

# Plot
EPISODES_TO_PLOT = 5
PLOT_MEAN_OVER_ROUNDS = 5

# Possible ACTIONS
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
