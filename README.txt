This project is a simple Pacman game environment where Pacman collects pellets and avoids ghosts. The Pacman is controlled by an agent using the A* search algorithm to find the best moves.

Files:
- `pacman_environment.py`: Defines the game environment (grid, walls, pellets, and ghost behaviors).
- `agent.py`: Contains the A* search logic for Pacman's movements.
- `main.py`: Runs the game and tracks the agent's performance.

How to Run:
1. Install dependencies:
pip install -r requirements.txt
2. Run the main program:
python3 agent_random.py


Features:
- **Environment**: A grid with walls, pellets, and two ghosts.
- **Agent**: Uses A* search to plan moves.
- **Ghosts**: One moves randomly, and the other chases Pacman.

Customization:
- Modify the grid layout in `pacman_environment.py`.
- Adjust agent behavior in `agent.py`.

Requirements:
- Python 3.7+
- Gymnasium
- Pygame
- Numpy

Author:
Created by Jaydin Blake.
