import numpy as np

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def evaluate(agents, env, ep, step, n_eval=3):
    score_history = []
    for i in range(n_eval):
        obs = env.reset()
        score = 0
        done = False

        while not done:
            actions = agents.choose_action(obs, evaluate=True)
            obs_, reward, done = env.step(actions.values())

            obs = obs_
            score += sum(reward)
        score_history.append(score)

    avg_score = np.mean(score_history)
    print(f'Evaluation episode {ep} train steps {step}'
          f' average score {avg_score:.1f}')
    return avg_score