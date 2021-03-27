AMOUNT_OF_TRAINING_EPISODES = 100000
NAME_OF_FILES = '5_layer_multi_enemy_destroyer_train_rule_based_decaybigger'

# Hyper parameters
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

BATCH_SIZE = 128
GAMMA = 0.9
EPS = 1
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10
LEARNING_RATE = 0.0001


ROUNDS_MODEL_UPDATE = 5
ROUNDS_NR = 1
INCREASE_STEPS_AFTER_EPISODES = 5000
INCREASE_STEP_VALUE = 20

# Plot
EPISODES_TO_PLOT = 500
PLOT_MEAN_OVER_ROUNDS = 100
SIZE_Y_AXIS = 10

# Possible ACTIONS
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
