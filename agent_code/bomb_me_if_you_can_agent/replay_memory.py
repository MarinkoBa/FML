import random


class ReplayMemory():
    '''
    Servers to record training data from the game.
    Memory is used to create random Batches to train the Deep Q Network.
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, c_state, c_action, n_state, reward):
        # Expand memory list until capacity is reached
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (c_state, c_action, n_state, reward)

        # increase position index until capacity is reached, than start jump back to index 0, (like RingBuffer)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''

        Args:
            batch_size: Number of samples which should be selected randomly.

        Returns: Random picked Batch (List of samples) from the memory.

        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
