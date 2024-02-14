import random
from enum import Enum


class Action_Type(str, Enum):
    MOVE_UP = 'Move_Up',
    MOVE_RIGHT = 'Move_Right',
    MOVE_DOWN = 'Move_Down',
    MOVE_LEFT = 'Move_Left',
    HUNT = 'Hunt',
    GATHER = 'Gather',
    FISH = 'Fish'


class Action:
    def __init__(self):
        """
        success_probability: integer from 0 to 100, probability of the action succeeding
        success_reward: integer, reward of the action succeeding
        failure_reward: integer, reward of the action failing
        coordinates: coordinates of the state the action can be performed ins
        """

    def do(self):
        pass


class Safe_Action(Action):
    def __init__(self, reward, coordinates):
        super().__init__()
        self.reward = reward
        self.coordinates = coordinates


class Action_Move_Up(Safe_Action):
    def __init__(self, coordinates):
        super().__init__(-1, coordinates)

    def do(self):
        new_coordinates = (self.coordinates[0] - 1, self.coordinates[1])
        return self.reward, new_coordinates


class Action_Move_Down(Safe_Action):
    def __init__(self, coordinates):
        super().__init__(-1, coordinates)

    def do(self):
        new_coordinates = (self.coordinates[0] + 1, self.coordinates[1])
        return self.reward, new_coordinates


class Action_Move_Right(Safe_Action):
    def __init__(self, coordinates):
        super().__init__(-1, coordinates)

    def do(self):
        new_coordinates = (self.coordinates[0], self.coordinates[1] + 1)
        return self.reward, new_coordinates


class Action_Move_Left(Safe_Action):
    def __init__(self, coordinates):
        super().__init__(-1, coordinates)

    def do(self):
        new_coordinates = (self.coordinates[0], self.coordinates[1] - 1)
        return self.reward, new_coordinates


class Risky_Action(Action):
    def __init__(self, success_probability, success_reward, failure_reward, resources, coordinates):
        super().__init__()
        self.success_probability = success_probability
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.resources = resources
        self.coordinates = coordinates

    def do(self):
        if self.resources == 0:
            return -1, self.coordinates
        else:
            self.resources -= 1
            if random.randint(0, 100) <= self.success_probability:
                return self.success_reward, self.coordinates
            else:
                return self.failure_reward, self.coordinates


class Action_Fish(Risky_Action):
    def __init__(self, coordinates):
        super().__init__(70, 2.5, -1, 10, coordinates)


class Action_Hunt(Risky_Action):
    def __init__(self, coordinates):
        super().__init__(50, 5, -3, 10, coordinates)


class Action_Gather(Risky_Action):
    def __init__(self, coordinates):
        super().__init__(90, 1, -1, 10, coordinates)
