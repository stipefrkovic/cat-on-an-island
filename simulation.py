import os
from statistics import mean

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from environment.environment import Environment
from learning.algorithm_factory import Learning_Algorithm_Type, create_learning_algorithm


class Simulation:
    """
    Class that runs the simulation
    Takes the agent, the environment and a desired number of simulation for training(?) as input
    """

    def __init__(self):
        self.environment = Environment()
        self.experiments = 1000
        self.experiment_time = 1000
        self.learning_algorithms_types = [Learning_Algorithm_Type.RANDOM,
                                          Learning_Algorithm_Type.Q_LEARNING_GREEDY,
                                          Learning_Algorithm_Type.Q_LEARNING_EPSILON_GREEDY,
                                          Learning_Algorithm_Type.Q_LEARNING_UCB,
                                          Learning_Algorithm_Type.Q_LEARNING_OPTIMISTIC,
                                          Learning_Algorithm_Type.Q_LEARNING_BOLTZMANN,
                                          Learning_Algorithm_Type.SARSA_GREEDY,
                                          Learning_Algorithm_Type.SARSA_EPSILON_GREEDY,
                                          Learning_Algorithm_Type.SARSA_UCB,
                                          Learning_Algorithm_Type.SARSA_OPTIMISTIC,
                                          Learning_Algorithm_Type.SARSA_BOLTZMANN]

    def run(self):
        """
        Initialises the simulation
        Sets the agent's position to a designated starting position
        """
        learning_algorithms_results = {}
        for learning_algorithm_type in self.learning_algorithms_types:
            learning_algorithms_results[learning_algorithm_type] = []
            for experiment in range(self.experiments):
                environment = Environment()
                environment.create_states()
                environment.create_agent()
                learning_algorithm = create_learning_algorithm(learning_algorithm_type, environment)
                time = learning_algorithm.run(self.experiment_time)
                learning_algorithms_results[learning_algorithm_type].append(time)

        for learning_algorithm_type, results in learning_algorithms_results.items():
            print(learning_algorithm_type, np.average(results), np.std(results), np.var(results))

        with open('results.csv', 'w') as f:
            f.write(','.join(['time_steps', 'algorithm']) + '\n')
            for learning_algorithm_type in learning_algorithms_results.keys():
                for time_steps in learning_algorithms_results[learning_algorithm_type]:
                    f.write(','.join([str(time_steps), learning_algorithm_type]) + '\n')


def main():
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
