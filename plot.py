import numpy as np
import os
import matplotlib.pyplot as plt

n_agents = 2
n_bodies = 1

maddpg_scores = np.load(f'data/maddpg_scores_{n_agents}_agent_{n_bodies}_body.npy')
maddpg_steps = np.load(f'data/maddpg_steps_{n_agents}_agent_{n_bodies}_body.npy')

file_name = f'maddpg_scores_{n_agents}_agent_{n_bodies}_body.png'
file_path = os.path.join('plots', file_name)

N = len(maddpg_scores)
running_avg = np.empty(N)
for t in range(N):
    running_avg[t] = np.mean(
            maddpg_scores[max(0, t-100):(t+1)])

plt.plot(maddpg_steps/100, running_avg, label='Scores', color='blue', linestyle='-')
plt.title('Performance de MADDPG au fil du temps')
plt.xlabel('Épisodes')
plt.ylabel('Récompenses')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(file_path)