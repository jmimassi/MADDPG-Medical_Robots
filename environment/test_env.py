from .Environment import Environment
import numpy as np
import time


def main():
    env = Environment(n_agents=2, n_bodies=2)
    max_episodes = 100
    try:
        print('Starting')
        for e in range(max_episodes):
            env.reset()
            total_episode_reward = 0
            while True:
                actions = [env.action_space[agent.name].sample()
                           for agent in env.agents]
                _, reward, done = env.step(actions)
                env.render()
                time.sleep(0.1)
                total_episode_reward += np.sum(reward)

                if done:
                    break
            print('Episode {}\tAverage reward: {:.2f}'.format(
                e + 1, total_episode_reward))

    finally:
        print('Closing')
        env.close()


if __name__ == "__main__":
    main()
