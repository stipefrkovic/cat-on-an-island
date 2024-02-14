class Agent:
    """
    Class that represents the agent
    """

    def __init__(self, current_state):
        self.current_state = current_state
        self.previous_state = None
        self.hp = 100
        self.last_reward = None

    def do_action(self, state, reward):
        self.previous_state = self.current_state
        self.current_state = state
        self.hp += reward
        self.last_reward = reward
