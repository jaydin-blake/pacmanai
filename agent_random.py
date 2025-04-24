#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import pacman_enviorment as pacman
import heapq
import random

from pacman_enviorment import getDistance

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Agent:
    def __init__(self):
        self.action_space = 4
        self.pacmanPosition = [14, 9]

    def astar_action(self, env):
        """Perform A* search to find the best action."""
        self.grid = env.grid  # Sync grid state with the environment
        self.pacmanPosition = env.pacman_position.copy()

        # Perform A* search
        path = self.astar(self.pacmanPosition, env)

        # Return the first move on the path
        if len(path) > 1:
            # The first position is Pacman's current position
            next_position = path[1]
            return self.get_direction(self.pacmanPosition, next_position)
        else:
            # If no path is found, take a safe action towards the nearest pellet
            action = self.get_best_safe_action(env)
            return action

    def astar(self, start, env, max_depth=50):
        """A* search to find the best path while avoiding ghosts."""
        open_set = []
        heapq.heappush(open_set, (0, 0, start, []))
        came_from = {}
        g_score = {tuple(start): 0}

        while open_set:
            _, current_g, current, path = heapq.heappop(open_set)
            time_step = current_g

            if current_g > max_depth:
                continue

            # Extend the path
            path = path + [current]

            # Check if we reached a pellet
            if env.grid[current[0], current[1]] == 1:
                return path

            # Get ghost positions
            ghost_positions = self.get_ghost_positions(env)

            for neighbor in self.get_neighbors(current, env):
                # Skip positions where ghosts are currently located
                if tuple(neighbor) in ghost_positions:
                    continue

                tentative_g_score = current_g + 1  # Uniform cost
                neighbor_tuple = tuple(neighbor)

                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, env)
                    heapq.heappush(
                        open_set, (f_score, tentative_g_score, neighbor, path))

        return [start]  # Default to no movement if no path is found

    def heuristic(self, position, env):
        """Heuristic function considering distance to pellet and ghosts."""
        # Distance to the nearest pellet
        pellet_positions = np.argwhere(env.grid == 1)
        if pellet_positions.size == 0:
            pellet_distance = 0
        else:
            pellet_distance = min(getDistance(position, pellet.tolist())
                                  for pellet in pellet_positions)

        # Distance to ghosts
        ghost_positions = self.get_ghost_positions(env)
        ghost_distances = [getDistance(position, list(
            ghost_pos)) for ghost_pos in ghost_positions]
        if ghost_distances:
            min_ghost_distance = min(ghost_distances)
            # Apply a risk factor based on proximity
            if min_ghost_distance == 0:
                ghost_penalty = 1000  # High penalty for collision
            else:
                ghost_penalty = 100 / min_ghost_distance
        else:
            ghost_penalty = 0

        # Total heuristic
        return pellet_distance + ghost_penalty

    def get_ghost_positions(self, env):
        """Get current positions of the ghosts."""
        positions = set()
        positions.add(tuple(env.randy.position))
        positions.add(tuple(env.folly.position))
        return positions

    def get_neighbors(self, position, env):
        """Get valid neighbors for a position."""
        y, x = position
        neighbors = []
        max_y, max_x = env.grid.shape
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < max_y and 0 <= nx < max_x:
                if not env.walls[ny, nx]:
                    neighbors.append([ny, nx])
        return neighbors

    def get_direction(self, start, next_position):
        """Get the direction from start to next_position."""
        y_diff = next_position[0] - start[0]
        x_diff = next_position[1] - start[1]
        if y_diff == -1:
            return UP
        elif y_diff == 1:
            return DOWN
        elif x_diff == -1:
            return LEFT
        elif x_diff == 1:
            return RIGHT
        return None

    def get_best_safe_action(self, env):
        """Get the best action towards the nearest pellet while avoiding ghosts."""
        safe_actions = []
        ghost_positions = self.get_ghost_positions(env)
        pellet_positions = np.argwhere(env.grid == 1)
        if pellet_positions.size == 0:
            return None  # No pellets left

        # Find the nearest pellet
        nearest_pellet = min(pellet_positions, key=lambda p: getDistance(
            env.pacman_position, p.tolist()))
        best_distance = float('inf')
        best_action = None

        for action in [UP, DOWN, LEFT, RIGHT]:
            new_position = env.pacman_position.copy()
            if action == UP:
                new_position[0] -= 1
            elif action == DOWN:
                new_position[0] += 1
            elif action == LEFT:
                new_position[1] -= 1
            elif action == RIGHT:
                new_position[1] += 1

            max_y, max_x = env.grid.shape
            ny, nx = new_position

            if 0 <= ny < max_y and 0 <= nx < max_x:
                if not env.walls[ny, nx]:
                    if tuple(new_position) not in ghost_positions:
                        distance = getDistance(
                            new_position, nearest_pellet.tolist())
                        if distance < best_distance:
                            best_distance = distance
                            best_action = action
        return best_action


def main():
    render_mode = "human"  # Disable rendering for faster execution
    agent = Agent()
    env = pacman.PacmanEnv(render_mode=render_mode)
    num_runs = 50
    results = []

    for run in range(num_runs):
        observation, info = env.reset()
        terminated = truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            action = agent.astar_action(env)
            if action is not None:
                step_result = env.step(action)
                if step_result is None:
                    # An error occurred in step; terminate this run
                    print(f"Run {run + 1}: Error in env.step(action)")
                    break
                observation, reward, terminated, truncated, info = step_result
                total_reward += reward
                steps += 1
            else:
                # No valid moves available for Pacman
                print(f"Run {run + 1}: failed score: {total_reward}")
                print(f"Run {run + 1}: failed steps: {steps}")
                break
            if (terminated or truncated):
                print(f"Run {run + 1}: total_reward: {total_reward}")
                print(f"Run {run + 1}: steps: {steps}")
                break

        win = np.sum(env.grid) == 0  # Check if all pellets are eaten
        results.append({
            'run': run + 1,
            'score': total_reward,
            'steps': steps,
            'win': win,
        })

    # Calculate statistics
    total_scores = [result['score'] for result in results]
    total_steps = [result['steps'] for result in results]
    win_rates = [result['win'] for result in results]

    average_score = sum(total_scores) / num_runs
    average_steps = sum(total_steps) / num_runs
    win_rate = sum(win_rates) / num_runs * 100  # Convert to percentage

    # Print the statistics
    print(f"Average Score: {average_score}")
    print(f"Average Steps: {average_steps}")
    print(f"Win Rate: {win_rate}%")


if __name__ == "__main__":
    main()
