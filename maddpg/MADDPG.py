from .Agent import Agent
import os  

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env,
                 alpha=1e-4, beta=1e-3, fc1=64, fc2=64, gamma=0.95, tau=0.01,
                 chkpt_dir='./checkpoints'):
        self.agents = []

        chkpt_dir += f'/{env.max_num_agents}_agent_{env.max_num_bodies}_body'
        try:  
            os.mkdir(chkpt_dir)  
        except OSError as error:  
            print(error) 

        for agent_idx in range(n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                               n_actions[agent_idx], n_agents, agent_idx,
                               alpha=alpha, beta=beta, tau=tau, fc1=fc1,
                               fc2=fc2, chkpt_dir=chkpt_dir,
                               gamma=gamma, min_action=0,
                               max_action=4))

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('Loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, evaluate=False):
        actions = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action = agent.choose_action(raw_obs[agent_id], evaluate)
            actions[agent_id] = action
        return actions

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)
