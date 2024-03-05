import numpy as np
from maddpg import MADDPG
from maddpg import ReplayBuffer
from environment import Environment
from maddpg import obs_list_to_state_vector, evaluate

def train():
    n_agents=2
    n_bodies=1
    grid_size=15.0

    gamma=0.95
    alpha=1e-4
    beta=1e-3
    batch_size=1024

    REPLAY_BUFFER_MAX_SIZE = 1_000_000
    EVAL_INTERVAL = 1000
    SAVE_INTERVAL = 10000
    MAX_STEPS = 500_000


    env = Environment(n_agents=n_agents,n_bodies=n_bodies, grid_size=grid_size)
    env.reset()

    n_agents = env.max_num_agents
    actor_dims = []
    n_actions = []

    for agent in env.agents:
        actor_dims.append(env.observation_space[agent.name].shape[0])
        n_actions.append(env.action_space[agent.name].shape[0])

    critic_dims = sum(actor_dims) + sum(n_actions)

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, env=env, gamma=gamma, alpha=alpha, beta=beta)
    critic_dims = sum(actor_dims)

    memory = ReplayBuffer(REPLAY_BUFFER_MAX_SIZE, critic_dims, actor_dims, n_actions, n_agents, batch_size=batch_size)

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
            obs_, reward, done = env.step(actions.values())

            # env.render()
            # time.sleep(1)

            list_obs = list(obs.values())
            list_actions = list(actions.values())
            list_obs_ = list(obs_.values())

            state = obs_list_to_state_vector(list_obs)
            state_ = obs_list_to_state_vector(list_obs_)

            memory.store_transition(list_obs, state, list_actions, reward, list_obs_, state_, done*env.max_num_agents)

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            obs = obs_
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        if total_steps % SAVE_INTERVAL == 0:
            maddpg_agents.save_checkpoint()

        episode += 1

    np.save(f'data/maddpg_scores_{n_agents}_agent_{n_bodies}_body.npy', np.array(eval_scores))
    np.save(f'data/maddpg_steps_{n_agents}_agent_{n_bodies}_body.npy', np.array(eval_steps))


if __name__ == '__main__':
    train()