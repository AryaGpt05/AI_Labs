"""
CS-F407: Artificial Intelligence
Laboratory - Agents: Constructing a Goal-Based Agent using an LLM

Author: Arya Gupta
"""

from collections import deque
from typing import List, Tuple, Optional, Set

# Warehouse map from laboratory specification
DEFAULT_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]

class WarehouseEnvironment:
    """Represents the 2D grid warehouse environment."""
    
    def __init__(self, grid: List[str]):
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._locate('S')
        self.goal = self._locate('G')
        
    def _locate(self, symbol: str) -> Tuple[int, int]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == symbol:
                    return (r, c)
        raise ValueError(f"Symbol '{symbol}' not found in map.")
        
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != '#'

class GoalBasedWarehouseAgent:
    """
    Goal-Based Agent that formulates a search problem to find
    a collision-free path to the destination.
    """
    
    ACTIONS = {
        'Up': (-1, 0),
        'Down': (1, 0),
        'Left': (0, -1),
        'Right': (0, 1)
    }
    
    def __init__(self, env: WarehouseEnvironment):
        self.env = env
        self.current_state = env.start
        self.goal_state = env.goal
        
    def plan_path(self) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[str]], int]:
        """
        Executes Breadth-First Search (BFS) to find the shortest collision-free path.
        Returns: (path_positions, path_actions, states_expanded)
        """
        queue = deque([(self.current_state, [self.current_state], [])])
        visited: Set[Tuple[int, int]] = {self.current_state}
        states_expanded = 0
        
        while queue:
            curr_pos, path_pos, path_act = queue.popleft()
            states_expanded += 1
            
            if curr_pos == self.goal_state:
                return path_pos, path_act, states_expanded
                
            r, c = curr_pos
            for action_name, (dr, dc) in self.ACTIONS.items():
                next_pos = (r + dr, c + dc)
                if self.env.is_valid_position(next_pos) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path_pos + [next_pos], path_act + [action_name]))
                    
        return None, None, states_expanded

    def display_solution(self, path: List[Tuple[int, int]]):
        """Displays the map with the solution path overlaid."""
        display_grid = [list(row) for row in self.env.grid]
        for r, c in path:
            if display_grid[r][c] not in ('S', 'G'):
                display_grid[r][c] = '*'
        print("\nWarehouse Navigation Map (* indicates solution path):")
        for row in display_grid:
            print("".join(row))

def main():
    print("=" * 60)
    print("Goal-Based Warehouse Navigation Agent")
    print("=" * 60)
    
    env = WarehouseEnvironment(DEFAULT_MAP)
    agent = GoalBasedWarehouseAgent(env)
    
    print(f"Start Position (S): {env.start}")
    print(f"Goal Position  (G): {env.goal}")
    print(f"Warehouse Dimensions: {env.rows} rows x {env.cols} columns")
    
    path_pos, path_act, states_expanded = agent.plan_path()
    
    if path_pos:
        print(f"\nStatus: Collision-free path found successfully!")
        print(f"Path Length (steps): {len(path_act)}")
        print(f"States Expanded: {states_expanded}")
        print(f"\nAction Sequence:\n{' -> '.join(path_act)}")
        print(f"\nCoordinate Trajectory:\n{path_pos}")
        agent.display_solution(path_pos)
    else:
        print("\nStatus: No collision-free path exists to the goal.")

if __name__ == '__main__':
    main()
