from maze_env import Maze
from QLearning import QL

def update():
    for t in range(100):
        observation = env.reset()
        # q.extendQtable(str(observation))
        k = 0
        while True:
            env.render()

            action = q.epsilonGreedy(str(observation))

            observation_, reward, done = env.step(int(action))

            # q.extendQtable(str(observation_))

            action_ = q.epsilonGreedy(str(observation_))

            q.updateValueQtable(str(observation), action, str(observation_), action_, reward )
            
            observation = observation_

            k += 1
            if done:
                print(f"this finish in {k} times")
                break



if __name__ == '__main__':
    env = Maze()
    q = QL([str(i) for i in range(env.n_actions)], 0.9, 0.9, 0.9)
    env.after(100, update)
    env.mainloop()