import gymnasium as gym
import numpy as np
from gymnasium import spaces
import heapq
import pygame
import random

# Constants for directions
UP = [-1, 0]
DOWN = [1, 0]
LEFT = [0, -1]
RIGHT = [0, 1]

# Colors and rendering settings
CELL_SIZE = 20
PACMAN_COLOR = (255, 255, 0)
FOOD_COLOR = (0, 255, 0)
WALL_COLOR = (52, 93, 169)
BG_COLOR = (0, 0, 0)
RANDY_COLOR = (235, 64, 52)   # Red
FOLLY_COLOR = (235, 52, 225)  # Pink

# Initial pacman_map representing the static layout
pacman_map = [
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
        '#', '#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', '0', '0', '0', '0', '0', '0', '0', '0', '#',
        '0', '0', '0', '0', '0', '0', '0', '0', '#'],
    ['#', '0', '#', '#', '0', '#', '#', '#', '0', '#',
        '0', '#', '#', '#', '0', '#', '#', '0', '#'],
    ['#', '0', '0', '0', '0', '0', '0', '0', '0', '0',
        '0', '0', '0', '0', '0', '0', '0', '0', '#'],
    ['#', '0', '#', '#', '0', '#', '0', '#', '#', '#',
        '#', '#', '0', '#', '0', '#', '#', '0', '#'],
    ['#', '0', '0', '0', '0', '#', '0', '0', '0', '#',
        '0', '0', '0', '#', '0', '0', '0', '0', '#'],
    ['#', '#', '#', '#', '0', '#', '#', '#', '0', '#',
        '0', '#', '#', '#', '0', '#', '#', '#', '#'],
    ['#', '#', '#', '#', '0', '#', '0', '0', '0', '0',
        '0', '0', '0', '#', '0', '#', '#', '#', '#'],
    ['#', '#', '#', '#', '0', '#', '0', '#', '#', '0',
        '#', '#', '0', '#', '0', '#', '#', '#', '#'],
    # MIDDLE
    ['#', '#', '#', '#', '0', '0', '0', '#', '0', '0',
        '0', '#', '0', '0', '0', '#', '#', '#', '#'],

    ['#', '#', '#', '#', '0', '#', '0', '#', '#', '#',
        '#', '#', '0', '#', '0', '#', '#', '#', '#'],

    ['#', '#', '#', '#', '0', '#', '0', '0', '0', '0',
        '0', '0', '0', '#', '0', '#', '#', '#', '#'],

    ['#', '#', '#', '#', '0', '#', '0', '#', '#', '#',
        '#', '#', '0', '#', '0', '#', '#', '#', '#'],
    ['#', '0', '0', '0', '0', '0', '0', '0', '0', '#',
        '0', '0', '0', '0', '0', '0', '0', '0', '#'],
    ['#', '0', '#', '#', '0', '#', '#', '#', '0', '#',
        '0', '#', '#', '#', '0', '#', '#', '0', '#'],
    ['#', '0', '0', '#', '0', '0', '0', '0', '0', '0',
        '0', '0', '0', '0', '0', '#', '0', '0', '#'],
    ['#', '#', '0', '#', '0', '#', '0', '#', '#', '#',
        '#', '#', '0', '#', '0', '#', '0', '#', '#'],
    ['#', '0', '0', '0', '0', '#', '0', '0', '0', '#',
        '0', '0', '0', '#', '0', '0', '0', '0', '#'],
    ['#', '0', '#', '#', '#', '#', '#', '#', '0', '#',
        '0', '#', '#', '#', '#', '#', '#', '0', '#'],
    ['#', '0', '0', '0', '0', '0', '0', '0', '0', '0',
        '0', '0', '0', '0', '0', '0', '0', '0', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
        '#', '#', '#', '#', '#', '#', '#', '#', '#'],


]


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        # 4 directions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(pacman_map), len(pacman_map[0])), dtype=np.int8
        )

        # Pacman's starting position
        self.pacman_position = [14, 9]
        # Initialize ghosts with references to the environment
        self.randy = RandyGhost([9, 10], self)
        self.folly = FollyGhost([9, 8], self)

        # Initialize grid and walls
        self.initialize_grid()

        # Pygame setup
        self.window_surface = None
        self.clock = None

    def initialize_grid(self):
        # Load grid layout
        self.grid = np.array(
            [[1 if cell == '0' else 0 for cell in row]
             for row in pacman_map], dtype=np.int8
        )
        self.walls = np.array(
            [[cell == '#' for cell in row] for row in pacman_map], dtype=bool
        )

        # Exclude the middle 3x3 cells where the ghosts spawn
        self.grid[9, 8:11] = 0
        self.grid[10, 8:11] = 0
        self.grid[11, 8:11] = 0

        # Remove walls at the entrance of the ghost house
        self.walls[10, 9] = False  # Entrance

        # Ensure no pellet at the entrance (optional)
        self.grid[10, 9] = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset Pacman's position
        self.pacman_position = [14, 9]
        # Reset ghosts' positions
        self.randy.position = [9, 10]
        self.folly.position = [9, 8]
        # Reinitialize the grid and walls
        self.initialize_grid()

        observation = self.grid.copy()
        info = {}
        return observation, info

    def step(self, action):
        try:
            # Determine new position
            new_position = self.pacman_position.copy()
            if action == 0:  # Up
                new_position[0] -= 1
            elif action == 1:  # Down
                new_position[0] += 1
            elif action == 2:  # Left
                new_position[1] -= 1
            elif action == 3:  # Right
                new_position[1] += 1

            max_y, max_x = self.grid.shape
            ny, nx = new_position

            # Check for wall collision and boundaries
            if 0 <= ny < max_y and 0 <= nx < max_x:
                if not self.walls[ny, nx]:
                    self.pacman_position = new_position

            # Eat pellet if present
            reward = 0
            if self.grid[tuple(self.pacman_position)] == 1:
                reward = 1
                # Pacman eats the pellet
                self.grid[tuple(self.pacman_position)] = 0

            # Update ghosts
            self.randy.step()  # Randy now moves randomly
            # Folly still needs Pacman's position
            self.folly.step(self.pacman_position)

            # Check for collisions with ghosts
            if self.pacman_position == self.randy.position or self.pacman_position == self.folly.position:
                terminated = True  # Pacman is caught
                reward = -10  # Penalty for being caught
            else:
                # End if all pellets are eaten
                terminated = np.sum(self.grid) == 0

            observation = self.grid.copy()
            info = {}

            if self.render_mode == "human":
                self.render()

            return observation, reward, terminated, False, info

        except Exception as e:
            print(f"An error occurred in step: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render(self):
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            window_width = CELL_SIZE * self.grid.shape[1]
            window_height = CELL_SIZE * self.grid.shape[0]
            self.window_surface = pygame.display.set_mode(
                (window_width, window_height))
            pygame.display.set_caption("Pacman")
            self.clock = pygame.time.Clock()

        self.window_surface.fill(BG_COLOR)

        # Draw walls, pellets, and Pacman
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.walls[y, x]:
                    pygame.draw.rect(
                        self.window_surface,
                        WALL_COLOR,
                        pygame.Rect(x * CELL_SIZE, y * CELL_SIZE,
                                    CELL_SIZE, CELL_SIZE),
                    )
                elif self.grid[y, x] == 1:
                    pygame.draw.circle(
                        self.window_surface,
                        FOOD_COLOR,
                        (x * CELL_SIZE + CELL_SIZE // 2,
                         y * CELL_SIZE + CELL_SIZE // 2),
                        CELL_SIZE // 8,
                    )

        # Draw Pacman
        pygame.draw.circle(
            self.window_surface,
            PACMAN_COLOR,
            (
                self.pacman_position[1] * CELL_SIZE + CELL_SIZE // 2,
                self.pacman_position[0] * CELL_SIZE + CELL_SIZE // 2,
            ),
            CELL_SIZE // 3,
        )

        # Draw ghosts
        self.randy.draw(self.window_surface)
        self.folly.draw(self.window_surface)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()


class RandyGhost:
    def __init__(self, position, env):
        self.position = position.copy()
        self.env = env  # Reference to the environment

    def draw(self, window_surface):
        # Drawing code remains the same
        pygame.draw.ellipse(
            window_surface,
            RANDY_COLOR,
            pygame.Rect(
                self.position[1] * CELL_SIZE + CELL_SIZE // 8,
                self.position[0] * CELL_SIZE + CELL_SIZE // 8,
                CELL_SIZE * 3 // 4,
                CELL_SIZE * 3 // 4,
            ),
        )
        pygame.draw.rect(
            window_surface,
            RANDY_COLOR,
            pygame.Rect(
                self.position[1] * CELL_SIZE + CELL_SIZE // 8,
                self.position[0] * CELL_SIZE + CELL_SIZE // 2,
                CELL_SIZE * 3 // 4,
                CELL_SIZE // 4,
            ),
        )
        # Eyes
        pygame.draw.circle(
            window_surface,
            (255, 255, 255),  # White eyes
            (
                self.position[1] * CELL_SIZE + CELL_SIZE * 0.3,
                self.position[0] * CELL_SIZE + CELL_SIZE * 0.4,
            ),
            CELL_SIZE // 10,
        )
        pygame.draw.circle(
            window_surface,
            (255, 255, 255),  # White eyes
            (
                self.position[1] * CELL_SIZE + CELL_SIZE * 0.7,
                self.position[0] * CELL_SIZE + CELL_SIZE * 0.4,
            ),
            CELL_SIZE // 10,
        )

    def step(self):
        y, x = self.position
        actions = self.get_actions(self.position)
        if not actions:
            return

        # Choose a random valid action
        action = random.choice(actions)
        new_y, new_x = y + action[0], x + action[1]

        # Update position
        self.position = [new_y, new_x]

    def get_actions(self, position):
        y, x = position
        actions = []
        max_y, max_x = self.env.grid.shape

        # Possible moves
        possible_moves = [UP, DOWN, LEFT, RIGHT]
        for move in possible_moves:
            new_y, new_x = y + move[0], x + move[1]
            if 0 <= new_y < max_y and 0 <= new_x < max_x:
                if not self.env.walls[new_y, new_x]:
                    actions.append(move)
        return actions


class FollyGhost:
    def __init__(self, position, env):
        self.position = position.copy()
        self.env = env  # Reference to the environment

    def draw(self, window_surface):
        # Drawing code remains the same
        pygame.draw.ellipse(
            window_surface,
            FOLLY_COLOR,
            pygame.Rect(
                self.position[1] * CELL_SIZE + CELL_SIZE // 8,
                self.position[0] * CELL_SIZE + CELL_SIZE // 8,
                CELL_SIZE * 3 // 4,
                CELL_SIZE * 3 // 4,
            ),
        )
        pygame.draw.rect(
            window_surface,
            FOLLY_COLOR,
            pygame.Rect(
                self.position[1] * CELL_SIZE + CELL_SIZE // 8,
                self.position[0] * CELL_SIZE + CELL_SIZE // 2,
                CELL_SIZE * 3 // 4,
                CELL_SIZE // 4,
            ),
        )
        # Eyes
        pygame.draw.circle(
            window_surface,
            (255, 255, 255),  # White eyes
            (
                self.position[1] * CELL_SIZE + CELL_SIZE * 0.3,
                self.position[0] * CELL_SIZE + CELL_SIZE * 0.4,
            ),
            CELL_SIZE // 10,
        )
        pygame.draw.circle(
            window_surface,
            (255, 255, 255),  # White eyes
            (
                self.position[1] * CELL_SIZE + CELL_SIZE * 0.7,
                self.position[0] * CELL_SIZE + CELL_SIZE * 0.4,
            ),
            CELL_SIZE // 10,
        )

    def step(self, pacman_position):
        # Folly uses A* but with a lookahead limit
        path = self.astar(self.position, pacman_position, max_depth=2)
        if path and len(path) > 1:
            # Move to the next position in the path
            self.position = path[1]

    def astar(self, start, goal, max_depth):
        """Perform A* search with a maximum depth (lookahead)."""
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        cost_so_far = {tuple(start): 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)
            if current == goal or current_cost >= max_depth:
                return self.reconstruct_path(came_from, start, current)

            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[tuple(current)] + 1
                if tuple(neighbor) not in cost_so_far or new_cost < cost_so_far[tuple(neighbor)]:
                    cost_so_far[tuple(neighbor)] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[tuple(neighbor)] = current

        return None  # No path found within max_depth

    def reconstruct_path(self, came_from, start, current):
        path = [current]
        while current != start:
            current = came_from[tuple(current)]
            path.append(current)
        path.reverse()
        return path

    def get_neighbors(self, position):
        y, x = position
        neighbors = []
        max_y, max_x = self.env.grid.shape
        moves = [UP, DOWN, LEFT, RIGHT]
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < max_y and 0 <= nx < max_x and not self.env.walls[ny, nx]:
                neighbors.append([ny, nx])
        return neighbors

    def heuristic(self, pos, goal):
        # Use Manhattan distance as heuristic
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def getDistance(position1, position2):
    """Calculate Manhattan distance between two positions."""
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])
