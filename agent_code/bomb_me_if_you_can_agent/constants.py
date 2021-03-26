BATCH_SIZE = 500
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200
TARGET_UPDATE = 10

EPS = 0.9

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


TRANSITIONS_AMOUNT_UPDATE = 10

ROUNDS_NR = 0


ROUNDS_MODEL_UPDATE = 5

# Plot
EPISODES_TO_PLOT = 1000
PLOT_MEAN_OVER_ROUNDS = 100


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...