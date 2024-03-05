import random
import numpy as np
from .utils import pixel_radius_to_grid_radius

class Body:
    def __init__(self, env, i) -> None:
        self.name = f'body_{i}'
        self.env = env
        self.radius = 30
        self.reward = 900/self.radius # Could be extended to list of different rewards
        self.health = 100
        self.position = self.simple_place_body(i) # Best for 15x15
        self.color = (0,255,0)


    def simple_place_body(self,i):
        POSITIONS = [(4,9),(10,4)] # Best for 15x15
        return POSITIONS[i]
    
    def random_place_body(self, i):
        # Taille de la grille
        grid_size = 15
        
        # Choix aléatoire des indices x et y
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        
        return (x, y)

    def complex_place_body(self,i):
        while True:
            x, y = np.random.uniform(1, self.env.grid_size - 1, size=2)
            new_position = np.array([x, y])

            too_close = False
            if len(self.env.bodies) > 1:
                for body in self.env.bodies:
                    radius1 = pixel_radius_to_grid_radius(self.radius,  self.env.grid_size, self.env.cell_size)
                    radius2 = pixel_radius_to_grid_radius(body.radius,  self.env.grid_size,  self.env.cell_size)
                    if np.linalg.norm(new_position - body.position) >= radius1 + radius2 :
                        too_close = True
                        break

            if not too_close:
                return new_position


    def update_health(self, healing= False):
        if not healing:
            self.health -= 1
