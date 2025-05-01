# pacmanai
#  Pacman AI Agent (Gymnasium)

This project implements an AI agent that plays Pacman using the Gymnasium framework. The agent navigates a grid-based maze to collect food pellets, avoid ghosts, and optimize its score based on predefined rewards and penalties.


https://github.com/user-attachments/assets/aa02fab2-765c-4502-a21e-aab387f7c3c9


## 🕹️ Environment

- **Framework:** [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- **Observation Space:** Grid matrix with information on:
  - Pacman position
  - Ghost positions
  - Food pellets
  - Walls
- **Action Space:**
  - `0` = UP
  - `1` = RIGHT
  - `2` = DOWN
  - `3` = LEFT

## 🎯 Agent Behavior

The AI agent uses a custom decision-making policy to:
- Collect food pellets (+10 reward)
- Avoid ghosts (-500 penalty if caught)
- Reach the goal (win condition)
- Learn or act based on a defined algorithm (e.g., A*, BFS, Q-Learning, etc.)

## 🧪 PEAS Description

- **Performance Measure:** Total score (food collected, time taken, ghost collisions)
- **Environment:** Grid maze with food, walls, and ghosts
- **Actuators:** Movement in four directions (up, right, down, left)
- **Sensors:** Agent perceives nearby grid content (walls, ghosts, food)

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
