from maze_env import Maze
from RL_brain import DeepQNetwork
import numpy as np
import time


def run_maze():
    since = time.time()
    step = 0
    for episode in range(3000):
        print("running maze "+str(episode)+" :3000")
        # initial observation
        observation = env.reset()
        # print(observation)
        # print("_________________")
        while True:
            # fresh env
            env.render()
            # print(observation)
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('run over')
    time_elapsed = time.time() - since
    # 代码计时
    print('The run_maze code run {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":

    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    run_maze()
    RL.plot_cost()