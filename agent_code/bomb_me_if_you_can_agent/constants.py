BATCH_SIZE = 500
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

EPS = 0.9

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']#, 'BOMB']


TRANSITIONS_AMOUNT_UPDATE = 50

ROUNDS_NR = 0


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...