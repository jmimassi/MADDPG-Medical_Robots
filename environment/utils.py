import random
import time
import numpy


def get_random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    color = (red, green, blue)
    if color != (0, 255, 0) and color != (200, 200, 200):
        return color
    get_random_color()


def set_color(colors):
    random_color = get_random_color()

    if random_color in colors:
        return set_color()
    else:
        return random_color


def display_tabular(environment, q_values, max_steps=100):
    '''
    q_values : (n, k) numpy array where n is the number of (discrete) states and k is the number of actions
    '''
    time_step = 0
    terminated, truncated = False, False
    state = environment.reset()
    while (not terminated and not truncated) and time_step < max_steps:
        environment.render()
        time.sleep(0.1)
        actions = []
        for i in range(environment.max_num_agents):
            x, y = state[f'agent_{i}']
            x, y = discretize(numpy.array([x, y]), numpy.array([0, 0]), numpy.array(
                [environment.grid_size, environment.grid_size]), [environment.grid_size**2, environment.grid_size**2])
            actions.append(discretize(find_index_of_max(
                q_values[i][x][y]), numpy.array([-1, -1]), numpy.array([1, 1]), [2, 2]))

        next_state, _, terminated, truncated = environment.step(actions)
        state = next_state
        time_step += 1
    environment.close()


def discretize(state, lower_bounds, upper_bounds, n_buckets):
    state = numpy.maximum(state, lower_bounds)
    state = numpy.minimum(state, upper_bounds)

    scaled = (state - lower_bounds) / (upper_bounds - lower_bounds)

    bucketed = scaled * (numpy.array(n_buckets) - 1)

    discrete_state = numpy.rint(bucketed).astype(int)

    return tuple(discrete_state)


def undiscretize(discrete_state, lower_bounds, upper_bounds, n_buckets):
    bucket_size = (upper_bounds - lower_bounds) / numpy.array(n_buckets)
    continuous_state = lower_bounds + \
        numpy.array(discrete_state) * bucket_size + bucket_size / 2

    return tuple(continuous_state)


def continuous_action(environment, discrete_action, action_bins):
    action_range = environment.action_space.high - environment.action_space.low
    bin_size = action_range / numpy.array(action_bins)
    continuous_action = environment.action_space.low + \
        bin_size * numpy.array(discrete_action) + bin_size / 2
    return continuous_action


def pixel_radius_to_grid_radius(pixel_radius, grid_size, cell_size):
    window_size = int(grid_size * cell_size)
    scale_factor = window_size / grid_size
    grid_radius = pixel_radius / scale_factor
    return grid_radius


def grid_radius_to_pixel_radius(grid_radius, grid_size, cell_size):
    window_size = int(grid_size * cell_size)
    scale_factor = window_size / grid_size
    pixel_radius = grid_radius * scale_factor
    return int(pixel_radius)


def find_index_of_max(matrix):
    index = numpy.unravel_index(numpy.argmax(matrix, axis=None), matrix.shape)
    return index
