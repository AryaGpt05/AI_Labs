"""
CS-F407: Artificial Intelligence
Laboratory - Search and A*: Using an LLM as an Engineering Assistant

Author: Arya Gupta
"""

import heapq
import math
from collections import deque
from typing import List, Tuple, Optional, Dict, Callable

ORIGINAL_WAREHOUSE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################"
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####"
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######"
]

ALT_PATHS_MAP = [
    "#######",
    "#S...G#",
    "#.###.#",
    "#.....#",
    "#######"
]


def find_position(grid: List[str], target: str) -> Optional[Tuple[int, int]]:
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == target:
                return (r, c)
    return None


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def zero_heuristic(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return 0.0


def a_star_search(
    grid: List[str],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Callable[[Tuple[int, int], Tuple[int, int]], float] = manhattan_distance,
    weight: float = 1.0
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[int], int]:
    """
    A* Search implementation.
    Returns: (path, path_length, states_expanded)
    """
    rows, cols = len(grid), len(grid[0])
    counter = 0
    # Priority queue entry: (f_score, g_score, tie_breaker, current_node, path)
    open_set = [(heuristic(start, goal) * weight, 0, counter, start, [start])]
    g_scores: Dict[Tuple[int, int], int] = {start: 0}
    states_expanded = 0

    while open_set:
        f, g, _, curr, path = heapq.heappop(open_set)
        states_expanded += 1

        if curr == goal:
            return path, g, states_expanded

        if g > g_scores.get(curr, float('inf')):
            continue

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                tentative_g = g + 1
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    counter += 1
                    h = heuristic(neighbor, goal) * weight
                    f_score = tentative_g + h
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor, path + [neighbor]))

    return None, None, states_expanded


def bfs_search(
    grid: List[str],
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[int], int]:
    """
    Breadth-First Search implementation.
    Returns: (path, path_length, states_expanded)
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start], 0)])
    visited = {start}
    states_expanded = 0

    while queue:
        curr, path, cost = queue.popleft()
        states_expanded += 1

        if curr == goal:
            return path, cost, states_expanded

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], cost + 1))

    return None, None, states_expanded


def render_path(grid: List[str], path: List[Tuple[int, int]]):
    display = [list(row) for row in grid]
    for r, c in path:
        if display[r][c] not in ('S', 'G'):
            display[r][c] = '*'
    for row in display:
        print("".join(row))


def run_all_tests():
    print("=" * 65)
    print("Laboratory 3: Search and A* Experimental Suite")
    print("=" * 65)

    # Test 1: Original Map
    print("\n--- Test 1: Original Warehouse Navigation ---")
    s1, g1 = find_position(ORIGINAL_WAREHOUSE, 'S'), find_position(ORIGINAL_WAREHOUSE, 'G')
    path_a, len_a, exp_a = a_star_search(ORIGINAL_WAREHOUSE, s1, g1, manhattan_distance)
    path_b, len_b, exp_b = bfs_search(ORIGINAL_WAREHOUSE, s1, g1)
    print(f"A* Search  | Path Found: {path_a is not None} | Length: {len_a} | Expanded: {exp_a}")
    print(f"BFS Search | Path Found: {path_b is not None} | Length: {len_b} | Expanded: {exp_b}")
    print("\nA* Solution Path Visualization:")
    render_path(ORIGINAL_WAREHOUSE, path_a)

    # Test 2: Trivial Map
    print("\n--- Test 2: Trivial Adjacent Map ---")
    s2, g2 = find_position(TRIVIAL_MAP, 'S'), find_position(TRIVIAL_MAP, 'G')
    p2, l2, e2 = a_star_search(TRIVIAL_MAP, s2, g2, manhattan_distance)
    print(f"A* | Path: {p2} | Length: {l2} | Expanded: {e2}")

    # Test 3: No Solution Map
    print("\n--- Test 3: Inaccessible Goal Map ---")
    s3, g3 = find_position(NO_SOLUTION_MAP, 'S'), find_position(NO_SOLUTION_MAP, 'G')
    p3, l3, e3 = a_star_search(NO_SOLUTION_MAP, s3, g3, manhattan_distance)
    print(f"A* | Path Found: {p3 is not None} | Length: {l3} | Expanded: {e3}")

    # Test 4: Alternative Paths
    print("\n--- Test 4: Alternative Paths Map ---")
    s4, g4 = find_position(ALT_PATHS_MAP, 'S'), find_position(ALT_PATHS_MAP, 'G')
    p4, l4, e4 = a_star_search(ALT_PATHS_MAP, s4, g4, manhattan_distance)
    print(f"A* | Length: {l4} | Expanded: {e4} (Optimal Shortest: {l4 == 4})")

    # Task 6: Heuristic Investigations
    print("\n" + "=" * 65)
    print("Task 6: Heuristic Variations on Original Warehouse")
    print("=" * 65)
    heuristics = [
        ("A* (Manhattan h)", manhattan_distance, 1.0),
        ("A* (Zero Heuristic h=0)", zero_heuristic, 1.0),
        ("A* (Euclidean Distance)", euclidean_distance, 1.0),
        ("A* (Inadmissible 2*Manhattan)", manhattan_distance, 2.0)
    ]

    print(f"{'Heuristic Strategy':<30} | {'Found?':<8} | {'Path Length':<12} | {'Expanded':<10}")
    print("-" * 65)
    for name, fn, w in heuristics:
        p, l, exp = a_star_search(ORIGINAL_WAREHOUSE, s1, g1, fn, weight=w)
        print(f"{name:<30} | {str(p is not None):<8} | {str(l):<12} | {exp:<10}")


if __name__ == '__main__':
    run_all_tests()
