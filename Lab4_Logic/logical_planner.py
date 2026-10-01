"""
CS-F407: Artificial Intelligence
Laboratory - Logical Reasoning for Planning

Author: Arya Gupta
"""

from collections import deque
from typing import List, Set, Tuple, Optional, Dict

class Action:
    """Represents a STRIPS-style planning action with logical preconditions and effects."""
    
    def __init__(
        self,
        name: str,
        pos_preconds: List[str],
        neg_preconds: List[str],
        pos_effects: List[str],
        neg_effects: List[str]
    ):
        self.name = name
        self.pos_preconds = set(pos_preconds)
        self.neg_preconds = set(neg_preconds)
        self.pos_effects = set(pos_effects)
        self.neg_effects = set(neg_effects)

    def is_applicable(self, state: Set[str]) -> bool:
        """Checks if S |= Preconditions(a)"""
        return self.pos_preconds.issubset(state) and len(self.neg_preconds.intersection(state)) == 0

    def apply(self, state: Set[str]) -> frozenset:
        """Computes successor state S' = (S \\ NegEffects) U PosEffects"""
        new_state = set(state)
        new_state.difference_update(self.neg_effects)
        new_state.update(self.pos_effects)
        return frozenset(new_state)


def plan_bfs(
    initial_state: Set[str],
    goal_state: Set[str],
    actions: List[Action]
) -> Tuple[Optional[List[str]], Optional[List[frozenset]]]:
    """
    Finds a plan (sequence of actions) from initial_state to goal_state using BFS.
    Returns: (action_sequence, state_sequence)
    """
    init = frozenset(initial_state)
    queue = deque([(init, [], [init])])
    visited = {init}

    while queue:
        curr_state, action_seq, state_seq = queue.popleft()

        # Goal test: Sn |= G
        if set(goal_state).issubset(curr_state):
            return action_seq, state_seq

        for action in actions:
            if action.is_applicable(curr_state):
                next_state = action.apply(curr_state)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((
                        next_state,
                        action_seq + [action.name],
                        state_seq + [next_state]
                    ))

    return None, None


def get_standard_actions(include_pickup: bool = True, include_move: bool = True, include_drop: bool = True) -> List[Action]:
    actions = []
    locations = ['A', 'B', 'C']
    connections = [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')]

    if include_move:
        for u, v in connections:
            actions.append(Action(
                name=f"Move({u}, {v})",
                pos_preconds=[f"At(Robot, {u})"],
                neg_preconds=[],
                pos_effects=[f"At(Robot, {v})"],
                neg_effects=[f"At(Robot, {u})"]
            ))

    if include_pickup:
        for loc in locations:
            actions.append(Action(
                name=f"PickUp(Package, {loc})",
                pos_preconds=[f"At(Robot, {loc})", f"At(Package, {loc})"],
                neg_preconds=["Holding(Package)"],
                pos_effects=["Holding(Package)"],
                neg_effects=[f"At(Package, {loc})"]
            ))

    if include_drop:
        for loc in locations:
            actions.append(Action(
                name=f"Drop(Package, {loc})",
                pos_preconds=[f"At(Robot, {loc})", "Holding(Package)"],
                neg_preconds=[],
                pos_effects=[f"At(Package, {loc})"],
                neg_effects=["Holding(Package)"]
            ))

    return actions


def run_tests():
    print("=" * 65)
    print("Laboratory 4: Logical Reasoning for Planning")
    print("=" * 65)

    # Test A: Solvable Problem
    print("\n--- Test A: Solvable Warehouse Delivery Problem ---")
    initial_A = {"At(Robot, A)", "At(Package, A)"}
    goal_A = {"At(Package, C)"}
    actions_A = get_standard_actions()

    act_seq_A, state_seq_A = plan_bfs(initial_A, goal_A, actions_A)
    if act_seq_A is not None:
        print("Plan Found Successfully!")
        print(f"Action Sequence: {' -> '.join(act_seq_A)}")
        print("\nState Progression:")
        for idx, (act, state) in enumerate(zip(["Initial State"] + act_seq_A, state_seq_A)):
            print(f"  Step {idx} [{act}]:\n    State = {sorted(list(state))}")
    else:
        print("No plan found.")

    # Test B: Impossible Problem (No PickUp)
    print("\n--- Test B: Impossible Problem (PickUp Action Removed) ---")
    actions_B = get_standard_actions(include_pickup=False)
    act_seq_B, _ = plan_bfs(initial_A, goal_A, actions_B)
    print(f"Result: {'No plan found' if act_seq_B is None else act_seq_B}")

    # Test C: Irrelevant Actions (Robot Moves without Package)
    print("\n--- Test C: Irrelevant Actions (Robot Moves Alone) ---")
    actions_C = get_standard_actions(include_pickup=False, include_drop=False)
    act_seq_C, _ = plan_bfs(initial_A, goal_A, actions_C)
    print(f"Result: {'No plan found' if act_seq_C is None else act_seq_C}")
    print("Verification: Planner correctly recognizes that At(Robot, C) != At(Package, C).")


if __name__ == '__main__':
    run_tests()
