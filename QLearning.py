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
        self.R = 0
        # self.Qtable = None
        # self.state = None
        # self.SList = []
        # self.policyList = []

    def reward(self,reward):
        """ reward function, which should be get value from the in surroundings."""
        self.R = reward
        return self.R

    def updateValueQtable(self, cur_state, action, next_state, Next_action):
        self.QTable.loc[cur_state, action] += self.alpha * (
                                            self.R + \
                                            self.gamma * self.QTable.loc[next_state, Next_action] - \
                                            self.QTable.loc[cur_state, action]
                                        )

    def stateExist(self,state):
        """check the state if it already exits in the Q Table."""
        if state in self.QTable.index:
            return True
        else:
            return False

    def extendQtable(self,state):
        if not self.stateExist(state):
            self.QTable = self.QTable.append(
                pd.Series(
                    [0]*len(self.actionSpace),
                    index = self.QTable.columns,
                    name = state,
                )
            )

    def epsilonGreedy(self, state):
        """epsilon greedy algorithm."""
        if rd.random() < self.epsilon:
            action = self.BestPolicy(state)
        else:
            action = np.random.choice(self.actionSpace)
        return action

    def BestPolicy(self, state):
        """ take an action into the environment"""
        if self.QTable.shape[0] == 0:
            return np.random.choice(self.actionSpace)
        actions = self.QTable.loc[state, :]
        actions = actions.reindex(np.random.permutation(actions.index))
        return actions.idxmax()

    def learning(self):
        pass

    def acquireState(self, state):
        """acquire state from environment."""
        pass

    def takeAction(self, env):
        pass


if __name__ == "__main__":
    pass