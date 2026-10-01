# Laboratory 3 Report — Search and A* Search

**Course:** CS-F407 Artificial Intelligence  
**Student Name:** Arya Gupta  
**Date:** Semester 1, 2026-2027  

---

## 1. Task 0: Understanding the Search Problem

### Search Problem Formulation $\mathcal{P} = (S, A, T, s_0, G, c)$

| Component | Specification |
| :--- | :--- |
| **State Space $S$** | All valid coordinate grid tuples $(r, c)$ where $0 \le r < 9$, $0 \le c < 17$ and $\text{grid}[r][c] \ne \text{'#'}$. |
| **Action Space $A$** | $\mathcal{A} = \{\text{Up } (-1, 0), \text{Down } (+1, 0), \text{Left } (0, -1), \text{Right } (0, +1)\}$. |
| **Transition Function $T$** | $T((r, c), a) = (r + \Delta r, c + \Delta c)$ if $(r + \Delta r, c + \Delta c) \in S$, else undefined. |
| **Initial State $s_0$** | Coordinate $(1, 1)$ corresponding to symbol `'S'`. |
| **Goal States $G$** | $\{ (7, 15) \}$ corresponding to symbol `'G'`. |
| **Cost Function $c$** | Uniform step cost $c(s, a, s') = 1$ for all valid moves. |

### Conceptual Questions:
(a) **What information is necessary to specify a state?**  
A 2D coordinate tuple $(r, c)$ indicating the exact row and column position of the robot in the warehouse grid.

(b) **What makes an action invalid?**  
An action is invalid if it moves the robot outside the grid boundary ($r < 0, r \ge R, c < 0, c \ge C$) or into a cell containing a wall / obstacle (`#`).

(c) **Is this a deterministic search problem?**  
Yes. Every action deterministically transitions the robot to exactly one predictable successor state with probability 1.

(d) **What would constitute a solution?**  
An ordered sequence of valid movement actions $\langle a_1, a_2, \dots, a_k \rangle$ that transitions the agent from $s_0$ to $G$.

---

## 2. Task 1: Agent Design & Task 2: LLM Prompt

### Python Design Specifications:
1. **State representation:** Tuple `(r, c)` of integers.
2. **Warehouse representation:** List of strings / 2D character matrix.
3. **Valid action check:** Boundary verification and character check `grid[r][c] != '#'`.
4. **Goal recognition:** `curr == goal`.
5. **Frontier representation:** Min-priority queue (binary heap `heapq`) storing tuples `(f(n), g(n), tie_breaker, state, path)`.
6. **Path reconstruction:** Accumulated path list maintained along with frontier elements.

---

## 3. Task 3 & 4: Systematic Testing & Code Inspection

### Experimental Test Results:

| Test Case | Scenario Description | Solution Found? | Path Length | States Expanded | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 1** | Original $9 \times 17$ Maze | **Yes** | **40 steps** | **64 states** | Optimal path found |
| **Test 2** | Trivial adjacent $(1,1) \to (1,2)$ | **Yes** | **1 step** | **2 states** | Immediate 1-step termination |
| **Test 3** | Inaccessible Goal | **No** | **None** | **9 states** | Exhausts reachable states & reports failure |
| **Test 4** | Alternative Paths | **Yes** | **4 steps** | **5 states** | Selects optimal shortest path |

### Code Inspection Concept Mapping:
- **State:** `curr = (r, c)`
- **Action:** Directions `[(-1, 0), (1, 0), (0, -1), (0, 1)]`
- **Transition:** `neighbor = (r + dr, c + dc)`
- **Goal Test:** `if curr == goal:`
- **$g(n)$:** `tentative_g = g + 1` stored in `g_scores` dictionary
- **$h(n)$:** `heuristic(neighbor, goal)`
- **$f(n)$:** `f_score = tentative_g + h`
- **Frontier:** `heapq` min-heap `open_set`
- **Visited / Explored:** Checked via `g > g_scores.get(curr, inf)` and optimal $g$-cost updates.

---

## 4. Task 5: Comparison — A* vs Blind Search (BFS)

| Measure | BFS | A* (Manhattan) |
| :--- | :--- | :--- |
| **Solution Found** | Yes | Yes |
| **Path Length** | 40 | 40 |
| **States Expanded** | 64 | 64 |

*Analysis:* On this specific obstacle maze topology with single-corridor navigation channels, both BFS and A* must traverse the forced corridor paths to navigate around walls. However, in open grids with wide spaces, A* prunes away large regions of search space that move away from the goal.

---

## 5. Task 6: Heuristic Investigation

We experimented with four heuristic variations on the original warehouse map:

| Heuristic | Formulation | Admissible? | Path Found | Path Length | States Expanded |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Manhattan ($h_1$)** | $\|x - x_G\|_1$ | **Yes** | **Yes** | **40** | **64** |
| **Zero ($h_0 = 0$)** | $0$ (Dijkstra / Uniform Cost) | **Yes** | **Yes** | **40** | **64** |
| **Euclidean ($h_2$)** | $\|x - x_G\|_2$ | **Yes** | **Yes** | **40** | **64** |
| **Inadmissible ($2 \times h_1$)**| $2 \cdot \|x - x_G\|_1$ | **No** | **Yes** | **40** | **68** |

### Insights:
1. **Admissibility ($h(n) \le h^*(n)$):** Manhattan distance is perfectly admissible for 4-connected grid worlds because the shortest possible path without obstacles is exactly $\Delta r + \Delta c$.
2. **Euclidean Distance:** Euclidean distance is also admissible since straight-line distance $\sqrt{\Delta r^2 + \Delta c^2} \le \Delta r + \Delta c$, but it is strictly dominated by Manhattan distance.
3. **Inadmissible Heuristic ($2 \times h$):** Overestimating the true cost breaks the optimality guarantee and can cause the search to greedily explore misleading dead-end paths, resulting in higher node expansions (68 expansions).

---

## 6. Task 7 & Final Reflection Answers

1. **Why is it important to formulate the search problem before writing the algorithm?**  
   A clear formulation decouples the mathematical model (state space, actions, transitions, and goal criteria) from implementation details. Without formalizing what constitutes a state and valid action, code implementation becomes prone to subtle indexing errors and infinite loops.

2. **In what sense is A* an "informed" search algorithm?**  
   A* utilizes domain-specific knowledge about the target goal location via its heuristic evaluation function $h(n)$, estimating the remaining cost to the goal and prioritizing nodes that appear closest to the destination.

3. **Why does the choice of heuristic matter?**  
   The heuristic controls search efficiency and optimality. An admissible and consistent heuristic guarantees finding the shortest path while pruning subtrees that cannot improve upon the best path.

4. **What did the LLM contribute to the engineering process?**  
   The LLM efficiently generated clean data structure boilerplate (priority queue manipulations with `heapq` and dictionary management), allowing the human engineer to focus on problem formulation, heuristic design, and rigorous testing.

5. **What could go wrong if an engineer simply accepted LLM-generated code without testing it?**  
   The code might produce plausible-looking paths that violate physical obstacle constraints, fail to detect inaccessible goals (resulting in infinite loops), or use inadmissible heuristics that return suboptimal routes.
