BATCH_SIZE = 60
GAMMA = 0.2
EPS_START = 0.2
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10

EPS = 0.9

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

TRANSITIONS_AMOUNT_UPDATE = 50
ROUNDS_MODEL_UPDATE = 5

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...