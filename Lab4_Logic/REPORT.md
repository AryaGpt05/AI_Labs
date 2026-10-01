# Laboratory 4 Report — Logical Reasoning for Planning

**Course:** CS-F407 Artificial Intelligence  
**Student Name:** Arya Gupta  
**Date:** Semester 1, 2026-2027  

---

## 1. Task 0: Understanding the Planning Problem

### Planning Problem Formulation $(I, A, G)$

* **Initial State $I$:**
  $$I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$$
* **Goal State $G$:**
  $$G = \{\text{At}(\text{Package}, C)\}$$
* **Actions Available:**
  1. $\text{Move}(u, v)$: Move robot between connected locations $u, v \in \{A, B, C\}$.
  2. $\text{PickUp}(\text{Package}, l)$: Robot picks up package at location $l$.
  3. $\text{Drop}(\text{Package}, l)$: Robot places package down at location $l$.

### Action Preconditions and Effects:

| Action | Positive Preconditions | Negative Preconditions | Positive Effects (Add List) | Negative Effects (Delete List) |
| :--- | :--- | :--- | :--- | :--- |
| $\text{Move}(u, v)$ | $\text{At}(\text{Robot}, u)$ | $\emptyset$ | $\text{At}(\text{Robot}, v)$ | $\text{At}(\text{Robot}, u)$ |
| $\text{PickUp}(\text{Package}, l)$ | $\text{At}(\text{Robot}, l), \text{At}(\text{Package}, l)$ | $\text{Holding}(\text{Package})$ | $\text{Holding}(\text{Package})$ | $\text{At}(\text{Package}, l)$ |
| $\text{Drop}(\text{Package}, l)$ | $\text{At}(\text{Robot}, l), \text{Holding}(\text{Package})$ | $\emptyset$ | $\text{At}(\text{Package}, l)$ | $\text{Holding}(\text{Package})$ |

### Initial Applicability Check:
* **Is $\text{PickUp}(\text{Package}, A)$ applicable in $I$?**  
  **Yes.** $\text{Preconditions} = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\} \subseteq I$. Both conditions are satisfied in $I$.
* **Is $\text{Drop}(\text{Package}, C)$ applicable in $I$?**  
  **No.** Preconditions require $\text{At}(\text{Robot}, C)$ and $\text{Holding}(\text{Package})$, neither of which are true in $I$.

---

## 2. Task 1: Manual Plan Construction

Step-by-step state progression:

| Step / Action | Current State Facts | Preconditions Satisfied? |
| :--- | :--- | :--- |
| **$S_0$ (Initial)** | $\{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$ | N/A |
| **$a_1 = \text{PickUp}(\text{Package}, A)$** | $\{\text{At}(\text{Robot}, A), \text{Holding}(\text{Package})\}$ | $\text{At}(\text{Robot}, A) \land \text{At}(\text{Package}, A)$ (True) |
| **$a_2 = \text{Move}(A, B)$** | $\{\text{At}(\text{Robot}, B), \text{Holding}(\text{Package})\}$ | $\text{At}(\text{Robot}, A)$ (True) |
| **$a_3 = \text{Move}(B, C)$** | $\{\text{At}(\text{Robot}, C), \text{Holding}(\text{Package})\}$ | $\text{At}(\text{Robot}, B)$ (True) |
| **$a_4 = \text{Drop}(\text{Package}, C)$** | $\{\text{At}(\text{Robot}, C), \text{At}(\text{Package}, C)\}$ | $\text{At}(\text{Robot}, C) \land \text{Holding}(\text{Package})$ (True) |

**Goal Verification:** $G = \{\text{At}(\text{Package}, C)\} \subseteq S_4$. The plan is valid and optimal.

---

## 3. Task 2 & 3: Implementation and Testing

### Prompt Given to LLM:
```text
Implement a STRIPS-like planning agent in Python.
Represent a state as a set of logical propositions.
Each action must have: name, positive preconditions, negative preconditions, positive effects, negative effects.
An action is applicable if positive preconditions are a subset of the state and negative preconditions have empty intersection.
Applying an action removes negative effects and adds positive effects.
Use BFS forward state space search to find the shortest plan to the goal.
Include full error handling for unsolvable problems.
```

### Test Results:

| Test Case | Configuration | Plan Found? | Resulting Plan | Validation |
| :--- | :--- | :--- | :--- | :--- |
| **Test A: Solvable** | Standard warehouse setup | **Yes** | `[PickUp(A), Move(A,B), Move(B,C), Drop(C)]` | 100% valid state progression |
| **Test B: Impossible**| `PickUp` action removed | **No** | `No plan found` | Correctly terminates without hallucinating moves |
| **Test C: Irrelevant**| Robot moves freely without package | **No** | `No plan found` | Distinguishes `At(Robot, C)` from `At(Package, C)` |

---

## 4. Task 4: Logic + Search = Planning

### Planning Workflow Diagram:
```
 Current State (S)
        ↓
 Check Action Preconditions (S |= Preconditions(a))
        ↓
 [ Filter Applicable Actions ]
        ↓
 Generate Successor State (S' = Apply(S, a))
        ↓
 Search over Alternatives (Frontier Queue / BFS)
        ↓
 Goal Test Satisfied? (Sn |= G)
```

**Core Principle:**  
*Logic determines what is physically and logically possible; search determines what sequence to explore to reach the objective.*

---

## 5. Task 6–8: Prolog as an Independent Logical Verifier

Using `planner.pl`:
* `?- can_move(a, b).` $\to$ **`true.`** (Direct fact matching).
* `?- can_move(a, c).` $\to$ **`false.`** (No direct connectivity fact between A and C).
* `?- valid_move(a, c).` $\to$ **`false.`** (Rejects invalid candidate actions proposed by an LLM).

### Task 8: Wet Road Logical Inference Chain:
$$\text{Fact}(\text{wet\_road}) \implies \text{Rule}(\text{wet\_road} \to \text{slippery}) \implies \text{Rule}(\text{slippery} \to \text{reduce\_speed}) \implies \text{Conclusion}(\text{reduce\_speed})$$

---

## 6. Answers to Reflection Questions

1. **Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**  
   It establishes formal operational semantics. Without strict preconditions and effects, an LLM might generate informal heuristic transitions that permit illegal teleports or violate physical conservation laws.

2. **Give an example of an error that could occur if the planner failed to check an action's preconditions.**  
   The robot could execute $\text{Drop}(\text{Package}, C)$ without ever picking up or carrying the package, or execute $\text{Move}(A, C)$ across disconnected locations.

3. **Why is a plan that "looks reasonable" not necessarily a valid plan?**  
   A plan may superficially mention all locations and actions in a plausible-looking story but skip critical prerequisite state transitions (such as picking up the package before moving).

4. **What did the LLM contribute to the implementation?**  
   The LLM efficiently wrote the Python class scaffolding, set algebra operations (`difference_update`, `update`), and BFS search loop.

5. **What did you have to verify independently?**  
   Ensured that state sets are treated as immutable frozensets for dictionary/set visited hashing, and verified that negative preconditions prevent double-pickup bugs.

6. **In this laboratory, where is logical reasoning being used?**  
   1. In testing precondition entailment ($S \models \text{Preconditions}(a)$).  
   2. In updating state beliefs via add and delete lists.  
   3. In goal satisfaction testing ($S_n \models G$).

7. **How is planning related to search algorithms?**  
   Planning is state-space search where states are structured logical models (sets of predicates) and transitions are discrete grounded operators dynamically instantiated by logical rules.
