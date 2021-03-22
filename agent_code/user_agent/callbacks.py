from agent_code.bomb_me_if_you_can_agent import features

def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']


