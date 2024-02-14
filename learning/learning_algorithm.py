class Learning_Algorithm:
    def __init__(self, environment):
        self.environment = environment

    def run(self, time):
        pass


class Q_Algorithm(Learning_Algorithm):
    def __init__(self, environment):
        super().__init__(environment)
        self.q_values = {}
        self.learning_rate = 0.5
        self.discount_factor = 0.9

    def get_best_action(self, state):
        return max(self.q_values[state], key=self.q_values[state].get)

    def select_action(self, state):
        raise Exception('Implement method.')

    def initialise_q_values(self, value):
        for state in self.environment.states.values():
            self.q_values[state] = {}
            for action in state.actions:
                self.q_values[state][action] = value

    def update_q_value(self, parameters):
        raise Exception('Implement method.')
