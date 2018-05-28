import gym
import time
from QLearning import QL
import matplotlib.pyplot as plt

def main():
    env = gym.make('CartPole-v0')
    q = QL([str(i) for i in range(env.action_space.n)], 0.95, 0.99, 0.6)
    plt.figure()

    for i_episode in range(100000):
        observation = env.reset()
        observation = [round(i,1) for i in observation]
        # q.extendQtable(str(observation))
        t = 0
        # for t in range(100):
        while True:

            env.render()
            action = q.epsilonGreedy(str(observation))
            observation_, reward, done, info = env.step(int(action))
            observation_ = [round(i,1) for i in observation_]
            # q.extendQtable(str(observation_))
            action_ = q.epsilonGreedy(str(observation_))
            if done == True :
                reward = -10
            else:
                reward = 2
            q.updateValueQtable(str(observation), action, str(observation_), action_, reward )

            observation = observation_
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                plt.plot(i_episode, t+1, 'g-*')
                break
    plt.show()

if __name__ == '__main__':
    main()