# Laboratory 2 Report — Goal-Based Intelligent Agents

**Course:** CS-F407 Artificial Intelligence  
**Student Name:** Arya Gupta  
**Date:** Semester 1, 2026-2027  

---

## 1. Task 1: Understanding the Problem

### 1. What is the environment?
The environment is a discrete, static, deterministic, and fully observable 2D grid world of dimensions $7 \times 21$. It contains:
- Obstacles / Shelving units denoted by `#` that block movement.
- Free traversal cells denoted by `.`.
- A starting location $S$ at $(1, 1)$.
- A destination / dispatch area $G$ at $(1, 19)$.

### 2. What is the goal of the agent?
The agent's objective is to compute and execute a collision-free sequence of moves that transitions the vehicle from its starting location $S$ to the destination $G$ while avoiding all shelving units.

### 3. What actions are available to the agent?
The agent has four discrete directional movement actions:
$$\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$$
Each valid action transitions the agent by $\pm 1$ grid unit in the vertical or horizontal axis with uniform unit step cost ($c = 1$).

### 4. What information must the agent maintain in order to choose its next action?
To make informed decisions, the agent must maintain:
- **Current state:** Its coordinate position $(r, c)$ on the grid.
- **Environment map:** Coordinates of all obstacles and traversable cells.
- **Goal specification:** The target coordinates $(r_G, c_G)$.
- **Search Frontier & Visited Set:** A record of explored states and active exploration paths to prevent infinite cycling in loops.

### 5. Why is this an example of a goal-based agent rather than a simple reflex agent?
- A **simple reflex agent** maps current percepts directly to actions via condition-action rules (e.g., *if obstacle ahead, turn right*). In a maze/warehouse with dead ends and concave obstacles, reflex agents easily get trapped in infinite loops.
- A **goal-based agent** combines knowledge of the current state, future goal state, and a model of how its actions affect the environment to formulate a multi-step plan before acting.

### Think About It: Scaling the Warehouse
If the warehouse becomes twice as large (e.g., $14 \times 42$ or higher):
- **Search Space Growth:** The number of states scales with the grid area ($O(N \cdot M)$), but the branching factor and search tree volume scale exponentially with path length ($b^d$).
- **Algorithm Suitability:** Uninformed search (BFS) becomes memory-intensive because it explores in concentric circles. An informed search strategy (such as $A^*$ with Manhattan heuristic) becomes essential to guide exploration directly toward the goal.

---

## 2. Task 2: Designing the Agent

### Component Architecture & Interaction:
```
 +-------------------------------------------------------+
 |                     ENVIRONMENT                       |
 |  - Grid Map (7 x 21)                                  |
 |  - Obstacles (#), Free Cells (.), Start (S), Goal (G) |
 +-------------------------------------------------------+
                            ^ |
                   Actions  | | Percepts (Current Position)
                            | v
 +-------------------------------------------------------+
 |                  GOAL-BASED AGENT                     |
 |                                                       |
 |  [1] State Tracker: Current State (r, c)              |
 |  [2] Transition Model: Next State = (r + dr, c + dc)   |
 |  [3] Goal Description: At(G)                          |
 |  [4] Decision / Planning Engine:                      |
 |      - Breadth-First Search / Shortest Path           |
 |      - Frontier Queue & Visited Set                   |
 |  [5] Action Execution Selector                        |
 +-------------------------------------------------------+
```

---

## 3. Task 3: Prompt Engineering & Implementation

### Prompt Provided to LLM:
```text
Write a clean, self-contained Python program implementing a Goal-Based Agent for the 2D warehouse navigation problem.
Map:
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################

Requirements:
- Represent the grid, obstacles, start (S), and goal (G).
- Formulate a goal-based search algorithm (BFS) to find the shortest collision-free path.
- Avoid all obstacles and grid boundary violations.
- Output the action sequence, coordinate trajectory, path length, and total states expanded.
- Render the map with the path overlaid.
```

### Questions on LLM Implementation:
1. **Did the LLM generate a working program on the first attempt?**  
   Yes. Because the prompt provided explicit map boundaries, coordinate specifications, and step-by-step requirements, the generated BFS code ran without syntax errors.
2. **If not, how can you improve your prompt?**  
   Providing explicit coordinate representations (0-indexed row/col) and clearly defining the termination condition prevented ambiguous boundary indexing.
3. **What search algorithm did the LLM choose?**  
   Breadth-First Search (BFS).
4. **Why did the LLM select this algorithm?**  
   Because all step costs are equal ($c=1$), BFS is guaranteed to find the optimal (shortest) path in terms of number of steps and explores systematically without getting stuck in infinite loops.

---

## 4. Execution & Validation Results

* **Start Coordinate:** `(1, 1)`
* **Goal Coordinate:** `(1, 19)`
* **Solution Found:** `True`
* **Path Length:** `20 steps`
* **States Expanded:** `59 states`
* **Action Sequence:**
  `Right -> Right -> Right -> Down -> Right -> Right -> Right -> Up -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right`
* **Visual Map Output:**
  ```text
  #####################
  #S***.#************G#
  #.##****##########..#
  #....##.............#
  #.######.###.#.###..#
  #........#..........#
  #####################
  ```
