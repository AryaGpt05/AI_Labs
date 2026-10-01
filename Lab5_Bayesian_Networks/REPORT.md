# Laboratory 5 Report — Bayesian Networks and Autoregressive Language Models

**Course:** CS-F407 Artificial Intelligence  
**Student Name:** Arya Gupta  
**Date:** Semester 1, 2026-2027  

---

## 1. Part I & II: Probability & Bayesian Network Structure

### Question 1: Why is the chain-rule decomposition useful for generating text?
The exact joint distribution over a full sentence $P(X_1, X_2, \dots, X_T)$ is practically impossible to estimate directly as a giant contingency table because the number of combinations grows exponentially with sentence length ($|V|^T$). The chain rule:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$$
decomposes the global joint probability into a sequence of tractable single-token conditional probability distributions. This enables left-to-right sequential text generation by sampling token $X_t$ conditioned on previously generated tokens $X_1, \dots, X_{t-1}$.

### Question 2: What independence assumption is made by the first-order network?
The first-order Markov model assumes that the probability of the current token $X_t$ depends *only* on the immediately preceding token $X_{t-1}$, conditionally independent of all earlier history:
$$P(X_t \mid X_1, X_2, \dots, X_{t-1}) = P(X_t \mid X_{t-1})$$
In Bayesian network notation:
$$X_t \perp\!\!\!\perp (X_1, \dots, X_{t-2}) \mid X_{t-1}$$

---

## 2. Part IV & VII: Conditional Probability Tables & Testing

### Question 3: Conditional Probability Table (1st-Order Model)

| Current Word ($X_{t-1}$) | Next Word ($X_t$) | Transition Count $C(w_i, w_j)$ | Conditional Probability $P(X_t \mid X_{t-1})$ |
| :--- | :--- | :--- | :--- |
| **`the`** | `cat` | 3 | $3/12 = 0.250$ |
| | `dog` | 3 | $3/12 = 0.250$ |
| | `mat` | 2 | $2/12 = 0.167$ |
| | `rug` | 2 | $2/12 = 0.167$ |
| | `park` | 2 | $2/12 = 0.167$ |
| **`cat`** | `sat` | 2 | $2/3 = 0.667$ |
| | `ran` | 1 | $1/3 = 0.333$ |
| **`dog`** | `sat` | 2 | $2/3 = 0.667$ |
| | `ran` | 1 | $1/3 = 0.333$ |
| **`sat`** | `on` | 4 | $4/4 = 1.000$ |
| **`ran`** | `to` | 2 | $2/2 = 1.000$ |
| **`<START>`** | `the` | 6 | $6/6 = 1.000$ |

*Zero-Probability Transitions:* $P(\text{sat} \mid \text{the}) = 0$, $P(\text{on} \mid \text{cat}) = 0$, $P(\text{the} \mid \text{mat}) = 0$, etc.

### Question 8: Normalization Test
For all valid words $w \in V$:
$$\sum_{v \in V} P(v \mid w) = 1.0000$$
If any row total evaluated to $0.87$, it would indicate an arithmetic normalizer bug in the divisor $\sum_k C(w_i, w_k)$ or unhandled out-of-vocabulary transitions.

---

## 3. Part VIII–X: Generation & Deterministic vs Probabilistic Comparison

### Question 9: Probability Predictions vs Human Linguistic Expectations
* **$\arg\max_w P(w \mid \text{the})$:** Returns `cat` (prob 0.25) or `dog` (prob 0.25).
* **Discussion:** A statistical probability model reflects only empirical frequencies in the training corpus without deep syntactic or semantic understanding. In contrast, human linguistic expectations enforce global grammatical agreement, discourse coherence, and world knowledge.

### Question 10: Greedy vs Sampling Generation
* **Greedy Generation (Mode A):** Always chooses $\arg\max_w P(w \mid w_{t-1})$. Because `the` transitions most frequently to `cat`/`dog` and `on` transitions deterministically to `the`, greedy generation enters an infinite cycle:
  `the cat sat on the cat sat on the cat sat on...`
* **Sampling Generation (Mode B):** Samples proportionally from $P(w \mid w_{t-1})$. Produces varied sentences with appropriate punctuation terminations when `<END>` is drawn.

---

## 4. Part XI–XIII: Second-Order Bayesian Networks

### Question 11: 1st-Order vs 2nd-Order Model Comparison

| Dimension | 1st-Order Model ($X_{t-1} \to X_t$) | 2nd-Order Model ($X_{t-2}, X_{t-1} \to X_t$) |
| :--- | :--- | :--- |
| **Graph Structure** | Chain with single parent | Directed graph with two parents per node |
| **CPT Condition** | $P(X_t \mid X_{t-1})$ | $P(X_t \mid X_{t-2}, X_{t-1})$ |
| **Context Window** | 1 previous token | 2 previous tokens |
| **Parameter Space** | $|V|^2$ entries | $|V|^3$ entries |
| **Data Requirement** | Moderate | Substantially higher (sparse bigram contexts) |

### Question 12: Context vs Data Scarcity Trade-off
Increasing context length from 1 to 2 enables the model to distinguish between `the` appearing at sentence start ($(\langle\text{START}\rangle, \text{the}) \to \text{cat/dog}$) versus `the` after a preposition $((\text{on}, \text{the}) \to \text{mat/rug}$; $(\text{to}, \text{the}) \to \text{park}$). However, the size of the CPT grows exponentially ($|V|^{n}$), causing data sparsity where unobserved tuples have zero counts without smoothing.

---

## 5. Part XIV–XV: Connection to Modern LLMs & Reflection

### Comparison with Modern Transformers:

| Feature | Simple Bayesian N-Gram | Modern Autoregressive LLM (GPT/Llama) |
| :--- | :--- | :--- |
| **Representation** | Explicit tabular CPTs | Deep neural attention weights |
| **Context** | 1 or 2 fixed tokens | Millions of tokens (multi-head self-attention)|
| **Parameters** | Observed discrete frequency counts | Billions of dense continuous weights |
| **Learning** | Maximum Likelihood Counting | Gradient descent backpropagation |
| **Generation** | Categorical sampling | Temperature / Top-$p$ nucleus sampling |

### Question 13: Prompting — Approach A vs Approach B
Approach B (*"Implement the following probabilistic model: $P(X_t \mid X_{t-1})$, estimated from transition counts..."*) is strictly superior because:
1. It establishes exact mathematical specifications rather than leaving architectural choices to LLM hallucination.
2. It allows explicit validation of probabilistic invariants ($\sum P = 1$).
3. It separates the underlying model definition from the computational implementation.

### Question 14: What did the Bayesian Network add?
1. **Factorisation:** Provided a principled mathematical justification for decomposing joint text probabilities into left-to-right conditional steps.
2. **Conditional Independence Reasoning:** Made explicit what assumptions are made when dropping distant context.
3. **Formal Verification:** Enabled systematic unit tests against probability axioms ($\sum_v P(v|w)=1$).
