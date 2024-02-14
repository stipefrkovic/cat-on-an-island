import numpy as np

from learning.learning_algorithm import Learning_Algorithm


class Random(Learning_Algorithm):
    def __init__(self, environment):
        super().__init__(environment)

    def select_action(self, state):
        return np.random.choice(state.actions)

    def update_q_value(self, parameters):
        pass

    def run(self, experiment_time):
        for time in range(experiment_time):
            action = self.select_action(self.environment.agent.current_state)
            if not self.environment.do_action(action):
                return time
        return experiment_time
