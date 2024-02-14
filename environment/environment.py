import random

from environment.action import Action, Action_Type, Action_Move_Up, Action_Move_Down, Action_Move_Left, \
    Action_Move_Right, Action_Fish, Action_Hunt, Action_Gather
from environment.agent import Agent
from environment.state import State, State_Type


class Environment:
    def __init__(self):
        """
        size: square root of the number of states in the environment
        forest_grass_ratio: ratio of the forest states to the grass states
        states: states in the environment
        agent: agent in the environment
        """
        self.size = 10
        self.forest_grass_ratio_percentage = 20
        self.beach_grass_ratio_percentage = 40
        self.states = {}
        self.agent = None
        self.time = 0

    def create_states(self):
        for i in range(self.size):
            for j in range(self.size):
                state = State((i, j))
                self.states[state.coordinates] = state
                if i != 0:
                    state.actions.append(Action_Move_Up(state.coordinates))
                if i != self.size - 1:
                    state.actions.append(Action_Move_Down(state.coordinates))
                if j != 0:
                    state.actions.append(Action_Move_Left(state.coordinates))
                if j != self.size - 1:
                    state.actions.append(Action_Move_Right(state.coordinates))

                if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                    if random.randint(0, 100) <= self.beach_grass_ratio_percentage:
                        state.type = State_Type.BEACH
                        state.actions.append(Action_Fish(state.coordinates))
                    else:
                        state.type = State_Type.GRASS
                else:
                    if random.randint(0, 100) <= self.forest_grass_ratio_percentage:
                        state.type = State_Type.FOREST
                        state.actions.append(Action_Hunt(state.coordinates))
                        state.actions.append(Action_Gather(state.coordinates))
                    else:
                        state.type = State_Type.GRASS

    def create_agent(self):
        self.agent = Agent(self.states[(0, 0)])

    def do_action(self, action):
        if action not in self.agent.current_state.actions:
            raise Exception('Illegal action ' + str(action.__class__.__name__) +
                            ' in state ' + str(self.agent.current_state.coordinates))
        else:
            reward, coordinates = action.do()
            self.agent.do_action(self.states[coordinates], reward)
            self.time += 1

            # f = open("report.txt", "a")
            # f.write("Agent is at state: " + str(self.agent.previous_state.coordinates) + "\n")
            # f.write(" and performs action: " + str(ac tion.__class__.__name__) + "\n")
            # f.write(" and receives reward: " + str(self.agent.last_reward) + "\n")
            # f.write(" and ends up in state: " + str(self.agent.current_state.coordinates) + "\n")
            # f.write(" and has HP: " + str(self.agent.hp) + "\n")
            # f.close()

            return self.agent.hp > 0

    def print(self):
        maze = [['🟩' for x in range(self.size)] for y in range(self.size)]
        for row in range(self.size):
            for column in range(self.size):
                if self.states[row, column].type == State_Type.FOREST:
                    # maze[row][column] = "\U0001F334"
                    # print("\U0001F334")
                    maze[row][column] = '🌴'
                elif self.states[row, column].type == State_Type.BEACH:
                    maze[row][column] = '🏖️'
        x = self.agent.current_state.coordinates[0]
        y = self.agent.current_state.coordinates[1]
        maze[x][y] = '🐈'
        for row in maze:
            print(' '.join(row))

