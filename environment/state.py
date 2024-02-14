from enum import Enum


class State_Type(str, Enum):
    BEACH = 'Beach',
    FOREST = 'Forest',
    GRASS = 'Grass'


class State:
    def __init__(self, coordinates):
        """
        coordinates: tuple with 2 elements: x and y position of the state
        type: State_Type
        actions: list with possible actions in the state
        """
        self.coordinates = coordinates
        self.type = None
        self.actions = []
