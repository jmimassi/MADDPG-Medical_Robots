import gym
from gym import spaces
import numpy as np
import pygame
from .Agent import Agent
from .Body import Body
from .utils import grid_radius_to_pixel_radius


class Environment(gym.Env):
    def __init__(self, max_timestep: int = 100, n_agents: int = 1, n_bodies: int = 1, grid_size: float = 15.0, cell_size: int = 20):
        super(Environment, self).__init__()
        self.cell_size = cell_size
        self.max_timestep = max_timestep
        self.max_num_agents = n_agents
        self.max_num_bodies = n_bodies
        self.grid_size = grid_size

        self.agents = []
        self.bodies = []

        max_distance = np.sqrt(grid_size**2 + grid_size**2)
        lower = np.concatenate(([0.0, 0.0, 0.0], [0.0 for _ in range(n_agents -1)]))
        upper = np.concatenate(([grid_size, grid_size, max_distance], [max_distance for _ in range(n_agents -1)]))
       
        self.observation_space = {f'agent_{i}': spaces.Box(low=lower, high=np.array(upper), dtype=np.float32) for i in range(n_agents)}
        self.action_space = {f'agent_{i}': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32) for i in range(n_agents)}

        self.viewer = None
        self.reset()

    def reset(self):
        self.timestep = 0
        self.state_history = []
        self.agents = [Agent(self, i) for i in range(self.max_num_agents)]
        self.bodies = [Body(self, i) for i in range(self.max_num_bodies)]
        self.rewards = []

        return self.get_state()

    def step(self, actions):
        self.timestep += 1
        rewards = []

        for i, action in enumerate(actions):
            old_pos = self.agents[i].position
            new_pos = self.agents[i].move(action)

            reward = self.simple_reward(old_pos, new_pos)
            self.agents[i].rewards.append(reward)
            rewards.append(reward)

        self.rewards.append(np.sum(reward))

        done = self.timestep >= self.max_timestep
        return self.get_state(), rewards, done

    def get_shape(self):
        return (self.max_num_agents, ) + (int(self.grid_size**2), int(self.grid_size**2)) + (2, 2)

    def get_state(self):
        state = {}
        for agent in self.agents:
            state[agent.name] = np.concatenate((
                agent.position,
                min([np.linalg.norm(np.array(agent.position) - np.array(body.position))] for body in self.bodies),
                agent.distance_to_other_agents()
                # ,agent.get_cumulative_reward()
            ))

            self.state_history.append(state)
        return state

    def simple_reward(self, old_position, new_position):
        within_radius = any(
            grid_radius_to_pixel_radius(
                np.linalg.norm(np.array(new_position) -
                               np.array(body.position)),
                self.grid_size, self.cell_size
            ) <= body.radius
            for body in self.bodies
        )

        return 1 if within_radius else -1

    def complex_reward(self, old_position, new_position):
        reward = 0

        within_radius = False
        for body in self.bodies:
            distance_to_body = np.linalg.norm(
                np.array(new_position) - np.array(body.position))

            reward = -distance_to_body
            distance_to_body = grid_radius_to_pixel_radius(
                distance_to_body, self.grid_size, self.cell_size)

            if distance_to_body <= body.radius:
                reward += body.reward
                within_radius = True
                break

        if not within_radius:
            distance_moved = np.linalg.norm(
                np.array(new_position) - np.array(old_position))

            if distance_moved <= 0.1:
                reward = -2
            else:
                reward = -1

        return reward

    def distance_reward(self, old_position, new_position):
        distances = []
        for body in self.bodies:
            distance_to_body = np.linalg.norm(
                np.array(new_position) - np.array(body.position))
            distances.append(distance_to_body)

        return - min(distances)

    def render(self):
        screen_size = int(self.grid_size * self.cell_size)

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_size, screen_size))

        self.viewer.fill((255, 255, 255))
        for agent in self.agents:
            for position in agent.positions_history:
                x, y = position
                pygame.draw.circle(self.viewer, agent.color,
                                   (int(x * self.cell_size), int(y * self.cell_size)), 2)
            x, y = agent.position
            pygame.draw.circle(self.viewer, agent.color,
                               (int(x * self.cell_size), int(y * self.cell_size)), 10)

        for body in self.bodies:
            x, y = body.position
            circle_center = (int(x * self.cell_size) + self.cell_size //
                             2, int(y * self.cell_size) + self.cell_size // 2)
            pygame.draw.circle(self.viewer, body.color,
                               circle_center, body.radius, 1)
            pygame.draw.rect(self.viewer, body.color, (int(
                x * self.cell_size), int(y * self.cell_size), self.cell_size, self.cell_size))

        pygame.display.flip()
