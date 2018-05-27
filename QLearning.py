import numpy as np
import pandas as pd

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

    def reward(self):
        """ reward function, which should be get value from the in surroundings."""
        pass

    def updateValueQtable(self):
        pass

    def stateExist(self):
        """check the state if it already exits in the Q Table."""
        pass

    def epsilonGreedy(self):
        """epsilon greedy algorithm."""
        pass

    def action(self, env):
        """ take an action into the envirnment"""
        pass

    def acquireState(self, state):
        """acquire state from enviornment."""
        pass