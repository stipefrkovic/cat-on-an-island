from enum import Enum

from learning.gradient_descent import Action_Preference
from learning.q_learning import Q_Learning_Greedy, Q_Learning_Epsilon_Greedy, Q_Learning_UCB, Q_Learning_Optimistic, Q_Learning_Boltzmann
from learning.random import Random
from learning.sarsa import Sarsa_Greedy, Sarsa_Epsilon_Greedy, Sarsa_UCB, Sarsa_Optimistic, Sarsa_Boltzmann


class Learning_Algorithm_Type(str, Enum):
    Q_LEARNING_GREEDY = 'Q_LEARNING_GREEDY'
    Q_LEARNING_EPSILON_GREEDY = 'Q_LEARNING_EPSILON_GREEDY',
    Q_LEARNING_UCB = 'Q_LEARNING_UCB',
    Q_LEARNING_OPTIMISTIC = 'Q_LEARNING_OPTIMISTIC',
    Q_LEARNING_BOLTZMANN = 'Q_LEARNING_BOLTZMANN'
    SARSA_GREEDY = 'SARSA_GREEDY',
    SARSA_EPSILON_GREEDY = 'SARSA_EPSILON_GREEDY',
    SARSA_UCB = 'SARSA_UCB',
    SARSA_OPTIMISTIC = 'SARSA_OPTIMISTIC',
    SARSA_BOLTZMANN = 'SARSA_BOLTZMANN'
    ACTION_PREFERENCE = 'ACTION_PREFERENCE',
    RANDOM = 'RANDOM'


def create_learning_algorithm(learning_algorithm_type, environment):
    if learning_algorithm_type is Learning_Algorithm_Type.Q_LEARNING_GREEDY:
        return Q_Learning_Greedy(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.Q_LEARNING_EPSILON_GREEDY:
        return Q_Learning_Epsilon_Greedy(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.Q_LEARNING_UCB:
        return Q_Learning_UCB(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.Q_LEARNING_OPTIMISTIC:
        return Q_Learning_Optimistic(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.Q_LEARNING_BOLTZMANN:
        return Q_Learning_Boltzmann(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.SARSA_GREEDY:
        return Sarsa_Greedy(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.SARSA_EPSILON_GREEDY:
        return Sarsa_Epsilon_Greedy(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.SARSA_UCB:
        return Sarsa_UCB(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.SARSA_OPTIMISTIC:
        return Sarsa_Optimistic(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.SARSA_BOLTZMANN:
        return Sarsa_Boltzmann(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.ACTION_PREFERENCE:
        return Action_Preference(environment)
    elif learning_algorithm_type is Learning_Algorithm_Type.RANDOM:
        return Random(environment)
    else:
        raise Exception('Illegal learning algorithm type.')
