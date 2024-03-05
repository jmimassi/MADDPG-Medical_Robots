import time
from maddpg import MADDPG
from environment import Environment
from maddpg import evaluate


def test():
    n_agents = 1
    n_bodies = 1

    gamma = 0.95
    alpha = 1e-4
    beta = 1e-3

    MAX_STEPS = 500_000

    env = Environment(n_agents=n_agents, n_bodies=n_bodies)
    env.reset()

    n_agents = env.max_num_agents
    actor_dims = []
    n_actions = []
    for agent in env.agents:
        actor_dims.append(env.observation_space[agent.name].shape[0])
        n_actions.append(env.action_space[agent.name].shape[0])
    critic_dims = sum(actor_dims) + sum(n_actions)

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents,
                           n_actions, env=env, gamma=gamma, alpha=alpha, beta=beta)
    maddpg_agents.load_checkpoint()

    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []

    score = evaluate(maddpg_agents, env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    while total_steps < MAX_STEPS:
        obs = env.reset()
        done = False

        while not done:
            actions = maddpg_agents.choose_action(obs)

            obs_, _, done = env.step(actions.values())

            env.render()
            time.sleep(0.01)
            obs = obs_


if __name__ == '__main__':
    test()
