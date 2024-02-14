import collections
import math

import numpy as np

from learning.learning_algorithm import Q_Algorithm


class Sarsa(Q_Algorithm):
    def __init__(self, environment):
        super().__init__(environment)

    def update_q_value(self, parameters):
        state = parameters.get('state')
        action = parameters.get('action')
        reward = parameters.get('reward')
        next_state = parameters.get('next_state')
        next_action = parameters.get('next_action')
        self.q_values[state][action] += self.learning_rate * (reward +
                                                              self.discount_factor * self.q_values[next_state][
                                                                  next_action]
                                                              - self.q_values[state][action])

    def run(self, experiment_time):
        action = self.select_action(self.environment.agent.current_state)
        for time in range(experiment_time):
            if not self.environment.do_action(action):
                return time
            else:
                next_action = self.select_action(self.environment.agent.current_state)
                parameters = {'state': self.environment.agent.previous_state,
                              'action': action,
                              'reward': self.environment.agent.last_reward,
                              'next_state': self.environment.agent.current_state,
                              'next_action': next_action,
                              'time': self.environment.time
                              }
                self.update_q_value(parameters)
                action = next_action
        return experiment_time


class Sarsa_Greedy(Sarsa):
    def __init__(self, environment):
        super().__init__(environment)
        self.initialise_q_values(0)

    def select_action(self, state):
        return self.get_best_action(state)


class Sarsa_Epsilon_Greedy(Sarsa):
    def __init__(self, environment):
        super().__init__(environment)
        self.initialise_q_values(0)
        self.epsilon = 0.05

    def select_action(self, state):
        p = np.random.random()
        if self.epsilon > p:
            return np.random.choice(state.actions)
        else:
            return self.get_best_action(state)


class Sarsa_UCB(Sarsa):
    def __init__(self, environment):
        super().__init__(environment)
        self.initialise_q_values(0)
        self.exploration_parameter = 0.2

        self.state_action_counter = {}
        self.u_values = {}
        for state in self.q_values.keys():
            self.state_action_counter[state] = {}
            self.u_values[state] = {}
            for action in self.q_values[state]:
                self.state_action_counter[state][action] = 0
                self.u_values[state][action] = self.exploration_parameter

    def select_action(self, state):
        counter = collections.Counter()
        for d in [self.q_values[state], self.u_values[state]]:
            counter.update(d)
        summed_q_u_values = dict(counter)
        action = max(summed_q_u_values, key=summed_q_u_values.get)
        return action

    def update_u_value(self, parameters):
        state = parameters.get('state')
        action = parameters.get('action')
        time = parameters.get('time')
        self.state_action_counter[state][action] += 1
        self.u_values[state][action] = self.exploration_parameter \
                                       * (math.sqrt(math.log(time) / self.state_action_counter[state][action]))

    def update_q_value(self, parameters):
        super().update_q_value(parameters)
        self.update_u_value(parameters)


class Sarsa_Optimistic(Sarsa):
    def __init__(self, environment):
        super().__init__(environment)
        self.initialise_q_values(10)

    def select_action(self, state):
        return self.get_best_action(state)


class Sarsa_Boltzmann(Sarsa):
    def __init__(self, environment):
        super().__init__(environment)
        self.initialise_q_values(0)
        self.temperature = 0.1

    def select_action(self, state):
        q_values = self.q_values[state]
        q_values = np.array(list(q_values.values()))
        q_values = q_values / self.temperature
        q_values = np.exp(q_values)
        q_values = q_values / np.sum(q_values)
        return np.random.choice(list(self.q_values[state].keys()), p=q_values)
