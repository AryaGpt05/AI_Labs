# CS-F407: Artificial Intelligence — Laboratory Coursework & Solutions

**Academic Year:** 2026–2027 (Semester 1)  
**Course Code:** CS-F407 (Artificial Intelligence)  
**Student Name:** Arya Gupta  

---

## 📖 Overview

This repository contains the complete laboratory implementations, standalone Python modules, Prolog verification rules, experimental results, and formal laboratory reports for **CS-F407 Artificial Intelligence**.

Each laboratory follows the foundational paradigm emphasized in the course:
$$\text{Understand} \longrightarrow \text{Design} \longrightarrow \text{Prompt / Implement} \longrightarrow \text{Execute \& Test} \longrightarrow \text{Reflect \& Verify}$$

The exercises bridge **AI Science** (formalizing problems, representation theorem, loss functions, graph formulations, probabilistic factorisations) with **AI Engineering** (automatic differentiation, convergence dynamics, heuristic design, numerical stability, and unit testing).

---

## 📂 Repository Structure

```
AI Labs/
├── .gitignore
├── README.md
├── requirements.txt
│
├── Lab1_Neural_Models/
│   ├── neur_models_lab_ex.pdf               # Original lab exercise specification
│   ├── neural_models.py                     # Standalone PyTorch experimental pipeline
│   └── REPORT.md                            # Comprehensive Markdown report with all tasks & reflection Qs
│
├── Lab2_Agents/
│   ├── agents_lab.pdf                       # Original lab exercise specification
│   ├── warehouse_agent.py                   # Goal-based warehouse navigation agent in Python
│   └── REPORT.md                            # Complete lab report & architectural breakdown
│
├── Lab3_Search/
│   ├── search_lab_ex.pdf                    # Original lab exercise specification
│   ├── search_agent.py                      # A* Search with multiple heuristics & BFS comparison
│   └── REPORT.md                            # Complete experimental evaluation & reflection answers
│
├── Lab4_Logic/
│   ├── logic_lab_ex.pdf                     # Original lab exercise specification
│   ├── logical_planner.py                   # STRIPS-style forward state-space BFS planning agent
│   ├── planner.pl                           # SWI-Prolog knowledge base & formal verifier rules
│   └── REPORT.md                            # Complete lab report & logic vs search analysis
│
├── Lab5_Bayesian_Networks/
│   ├── BN_lab.pdf                           # Original lab exercise specification
│   ├── language_model.py                    # 1st-order and 2nd-order autoregressive language models
│   ├── generated_sentences_first_order.txt  # 20 sampled text sequences from 1st-order model
│   ├── generated_sentences_second_order.txt # 20 sampled text sequences from 2nd-order model
│   └── REPORT.md                            # Complete report answering Questions 1–14
│
└── Lab6_Transformers/
    ├── transformers_demo.py                 # Transformer paradigms demonstration script
    └── REPORT.md                            # Theoretical synthesis & architectural analysis
```

---

## 🔬 Summary of Laboratory Modules

### [Lab 1: Neural Models — Learning, Depth, Activations, and Output Layers](./Lab1_Neural_Models/)
- **Core Concept:** Non-linear representation learning on XOR redundant safety sensor data.
- **Key Investigations:**
  - Proved mathematical necessity of hidden non-linearities: $\tilde{W}x + \tilde{b}$ cannot separate XOR.
  - Implemented 2–2–1 architecture in PyTorch with `BCEWithLogitsLoss`.
  - Analyzed the **symmetry problem** caused by zero-weight initialization ($\nabla_{W^{(1)}} L = 0$).
  - Evaluated **Sigmoid vs Tanh vs ReLU** activation dynamics and early gradient norms.
  - Extended model to **3-Class classification** with Softmax output and verified constant-shift invariance ($+100$).

### [Lab 2: Intelligent Agents — Goal-Based Navigation](./Lab2_Agents/)
- **Core Concept:** Goal-Based Agent architecture vs Simple Reflex Agent.
- **Key Investigations:**
  - Modeled a 2D $7 \times 21$ warehouse grid environment with obstacles and start/destination bays.
  - Designed internal world-state model, action transitions $\{\text{Up, Down, Left, Right}\}$, and goal-satisfaction rules.
  - Demonstrated why reflex agents get trapped in concave obstacles while goal-based BFS planners guarantee optimal collision-free trajectories (20 steps, 59 states expanded).

### [Lab 3: Search and A* Search](./Lab3_Search/)
- **Core Concept:** Informed heuristic search $\mathcal{P} = (S, A, T, s_0, G, c)$ and $f(n) = g(n) + h(n)$.
- **Key Investigations:**
  - Implemented A* search with min-priority queue and compared with BFS.
  - Validated against a 4-test suite (original maze, trivial 1-step, inaccessible goal, alternative paths).
  - Investigated heuristic properties: **Manhattan** (admissible, optimal), **Zero/Dijkstra** ($h=0$), **Euclidean** (admissible but strictly dominated), and **Inadmissible** ($2 \times \text{Manhattan}$, greedy expansion trap).

### [Lab 4: Logical Reasoning for Planning](./Lab4_Logic/)
- **Core Concept:** $\text{Logic} + \text{Search} = \text{Planning}$.
- **Key Investigations:**
  - Implemented STRIPS-style planning operator model in Python: preconditions $S \models \text{Precond}(a)$ and state updates $S' = (S \setminus \text{NegEff}) \cup \text{PosEff}$.
  - Successfully planned optimal action sequence: $\text{PickUp}(A) \to \text{Move}(A,B) \to \text{Move}(B,C) \to \text{Drop}(C)$.
  - Integrated **SWI-Prolog** (`planner.pl`) as an independent formal logic verifier to validate candidate plans against ground-truth connectivity rules.

### [Lab 5: Bayesian Networks and Autoregressive Language Models](./Lab5_Bayesian_Networks/)
- **Core Concept:** Chain-rule joint distribution factorisation $P(X_1 \dots X_T) = \prod P(X_t \mid X_{<t})$.
- **Key Investigations:**
  - Constructed conditional probability tables (CPTs) from corpus counts and verified probability normalisation $\sum_v P(v \mid w) = 1.0$.
  - Demonstrated that 1st-order greedy generation enters infinite repeating cycles, while 2nd-order Markov models preserve syntactic context and terminate cleanly.
  - Connected discrete n-gram Bayesian models to modern deep autoregressive Transformers.

### [Lab 6: Transformers & Modern Language Models](./Lab6_Transformers/)
- **Core Concept:** Scaled dot-product multi-head attention and transformer topologies.
- **Key Investigations:**
  - **Encoder-Decoder (T5 / FLAN-T5):** Cross-attention for sequence translation and text summarization.
  - **Decoder-Only (GPT-2):** Causal self-attention for generative autoregression.
  - **Encoder-Only (BERT):** Bidirectional masked language modeling for sentiment classification.

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.12)
- Git

### 2. Environment Installation
```bash
# Clone the repository
git clone https://github.com/AryaGpt05/CS-F407-AI-AY2026-27-S1.git
cd CS-F407-AI-AY2026-27-S1

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# or: .venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Laboratory Scripts
```bash
# Lab 1: Neural Models
python Lab1_Neural_Models/neural_models.py

# Lab 2: Agents
python Lab2_Agents/warehouse_agent.py

# Lab 3: Search & A*
python Lab3_Search/search_agent.py

# Lab 4: Logic Planning
python Lab4_Logic/logical_planner.py

# Lab 5: Bayesian Networks & Language Models
python Lab5_Bayesian_Networks/language_model.py

# Lab 6: Transformers
python Lab6_Transformers/transformers_demo.py
```

---
*Maintained by Arya Gupta for CS-F407 Artificial Intelligence.*
