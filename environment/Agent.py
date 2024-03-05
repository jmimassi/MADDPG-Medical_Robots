import numpy as np
from .utils import grid_radius_to_pixel_radius,  set_color

class Agent:
    def __init__(self, env, i) -> None:
        self.name = f'agent_{i}'
        self.env = env
        self.positions_history = []
        self.visited = {}
        self.position = self._place_agent()
        self.color = set_color([agent.color for agent in env.agents])
        self.rewards = []

    def _place_agent(self):
        x, y = np.random.uniform(1, self.env.grid_size - 1, size=2)
        return np.array([x, y])

    def move(self, action):
        dx, dy = action

        dx = -0.5 + dx
        dy = -0.5 + dy

        new_position = self.position + np.array([dx, dy])

        new_position = np.clip(new_position, 0, self.env.grid_size)
        self.positions_history.append(new_position)
        self.position = new_position
        return new_position

    def get_cumulative_reward(self):
        return np.sum(self.rewards)
    
    def is_in_radius(self):
        for body in self.env.bodies:
            distance_to_body = np.linalg.norm(
            np.array(self.position) - np.array(body.position))

            distance_to_body = grid_radius_to_pixel_radius(
                distance_to_body, self.env.grid_size, self.env.cell_size)
            if distance_to_body <= body.radius:
                return 1.0
        return 0.0

    def distance_to_other_agents(self):
        distances = []
        for other_agent in self.env.agents:
            if other_agent != self:  # Exclure l'agent actuel
                distance_to_other_agent = np.linalg.norm(
                    np.array(self.position) - np.array(other_agent.position)
                )
                distances.append(distance_to_other_agent)

        return distances