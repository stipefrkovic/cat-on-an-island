import random

import numpy as np

from learning.learning_algorithm import Learning_Algorithm


class Action_Preference(Learning_Algorithm):
    def __init__(self, environment):
        super().__init__(environment)
        self.learning_rate = 0.5
        self.action_index_histogram = {}
        self.action_reward_history = []
        self.action_index_history = []

        self.action_preferences = {}
        self.action_policy = {}
        self.action_average_reward = {}
        self.action_cumulative_reward = {}
        self.initialise_values()

    def initialise_values(self):
        for state in self.environment.states.values():
            self.action_preferences[state] = {}
            self.action_policy[state] = {}
            self.action_average_reward[state] = {}
            self.action_cumulative_reward[state] = 0
            for action in state.actions:
                self.action_preferences[state][action] = 0
                self.action_policy[state][action] = 0.01
                self.action_average_reward[state][action] = 0

    def select_action(self, state):
        policy_list = list(self.action_policy[state].values())
        actions = list(self.action_policy[state].keys())
        action = random.choices(actions, weights=policy_list, k=1)[0]
        return action

    def update_action_preferences(self, parameters):
        state = parameters.get('state')
        action = parameters.get('action')
        time = parameters.get('time')
        reward = parameters.get('reward')

        self.action_average_reward[state][action] = self.action_cumulative_reward[state] / time
        for a in state.actions:
            if a == action:
                self.action_preferences[state][action] += self.learning_rate \
                                                          * (reward - self.action_average_reward[state][action]) \
                                                          * (1 - self.action_policy[state][action])
            else:
                self.action_preferences[state][action] -= self.learning_rate \
                                                          * (reward - self.action_average_reward[state][action]) \
                                                          * self.action_policy[state][action]
        self.update_action_policy(state)

    def update_action_policy(self, state):
        exp_sum = 0
        for action in state.actions:
            exp_sum += np.exp(self.action_preferences[state][action])
        for action in state.actions:
            self.action_policy[state][action] = np.exp(self.action_preferences[state][action]) / exp_sum

    def run(self, time):
        for time in range(time):
            action = self.select_action(self.environment.agent.current_state)
            if not self.environment.do_action(action):
                return time
            else:
                parameters = {'state': self.environment.agent.previous_state,
                              'action': action,
                              'reward': self.environment.agent.last_reward,
                              'next_state': self.environment.agent.current_state,
                              'time': self.environment.time
                              }
                self.update_action_preferences(parameters)
        return time
