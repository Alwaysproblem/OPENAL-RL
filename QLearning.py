import numpy as np
import pandas as pd
import random as rd

class QL:
    def __init__(self, actionSpace, epsilon, gamma, alpha):
        self.actionSpace = actionSpace
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.QTable = pd.DataFrame(columns = self.actionSpace, dtype = np.float)
        # self.Qtable = None
        # self.state = None
        # self.SList = []
        # self.policyList = []

    def reward(self,reward):
        """ reward function, which should be get value from the in surroundings."""
        return reward

    def updateValueQtable(self,state):
        pass
        # if not self.stateExist(state):
        #     self.QTable = self.QTable.append(pd.Series(
        #             [0]*len(self.actionSpace),
        #             index=self.QTable.columns,
        #             name=state,
        #         )
        #     )

    def stateExist(self,state):
        """check the state if it already exits in the Q Table."""
        if state in self.QTable.index:
            return True
        else:
            return False

    def takeAction(self, env):
        pass

    def epsilonGreedy(self, state):
        """epsilon greedy algorithm."""
        if rd.random() < self.epsilon:
            action = self.BestPolicy(state)
        else:
            action = np.random.choice(self.actionSpace)
        return action

    def BestPolicy(self, state):
        """ take an action into the environment"""
        actions = self.QTable.loc[state, :]
        # actions.

    def acquireState(self, state):
        """acquire state from environment."""
        pass


if __name__ == "__main__":
    pass