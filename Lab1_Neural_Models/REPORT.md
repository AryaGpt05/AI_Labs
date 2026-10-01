# Laboratory 1 Report — Neural Models: Learning, Depth, Activations, and Output Layers

**Course:** CS-F407 Artificial Intelligence  
**Student Name:** Arya Gupta  
**Date:** Semester 1, 2026-2027  

---

## 1. Task 1: Problem Formulation & Linear Separability

### Problem Specification
We consider a redundant safety sensor scenario where two binary sensors, $x_1, x_2 \in \{0, 1\}$, monitor a system. An alarm / disagreement warning $y \in \{0, 1\}$ must be raised when exactly one sensor is active and the other is inactive.

* **Input Space:** $\mathcal{X} = \{0, 1\}^2 = \{(0,0), (0,1), (1,0), (1,1)\}$
* **Output Space:** $\mathcal{Y} = \{0, 1\}$
* **Labelled Dataset (XOR Truth Table):**
  - $(0, 0) \to 0$ (Both sensors inactive)
  - $(0, 1) \to 1$ (Sensor 2 active, Sensor 1 inactive)
  - $(1, 0) \to 1$ (Sensor 1 active, Sensor 2 inactive)
  - $(1, 1) \to 0$ (Both sensors active)

### Linear Separability Explanation
If we plot the four points in the $(x_1, x_2)$ 2D coordinate plane:
- Points $(0, 1)$ and $(1, 0)$ with label $y=1$ occupy the off-diagonal vertices.
- Points $(0, 0)$ and $(1, 1)$ with label $y=0$ occupy the main diagonal vertices.

Any single affine linear decision boundary is defined by $w_1 x_1 + w_2 x_2 + b = 0$. A straight line dividing the plane into two half-spaces cannot simultaneously group $\{(0,1), (1,0)\}$ on one side and $\{(0,0), (1,1)\}$ on the other. The convex hulls of the two classes intersect.

### Prediction for a Single Affine Transformation + Sigmoid Output
If we train a single affine transformation followed by a sigmoid $\hat{y} = \sigma(W x + b)$ on XOR:
- The model will fail to separate the classes.
- The loss will plateau near $\ln(2) \approx 0.693$ (the entropy of guessing uniform probability $p=0.5$).
- The linear model can at best achieve 75% accuracy by classifying 3 points correctly and misclassifying the 4th, or output $p \approx 0.5$ everywhere.

---

## 2. Task 2: Intelligent Agent Design & Validation Criteria

### Model Architecture
- **Structure:** $2 \text{ inputs} \to 2 \text{ hidden units} \to 1 \text{ output logit}$ (2–2–1).
- **Hidden Activation:** Non-linear activation ($\tanh$, $\text{Sigmoid}$, or $\text{ReLU}$).
- **Output Layer:** Linear logit paired with `BCEWithLogitsLoss` (numerically stable binary cross-entropy with built-in sigmoid).
- **Optimiser:** Adam ($\text{lr} = 0.05 - 0.08$) or SGD with momentum.

### Conceptual Justifications
1. **Why is hidden non-linearity scientifically necessary?**  
   Without non-linear hidden activations, a sequence of affine transformations collapses into a single affine transformation:
   $$W^{(2)}(W^{(1)}x + b^{(1)}) + b^{(2)} = (W^{(2)}W^{(1)})x + (W^{(2)}b^{(1)} + b^{(2)}) = \tilde{W}x + \tilde{b}$$
   Depth without non-linearity provides zero additional expressive capacity for non-linear decision boundaries. The non-linear activation warps the feature space, mapping non-linearly separable inputs into a linearly separable hidden representation $h \in \mathbb{R}^2$.

2. **Why is Sigmoid + Binary Cross-Entropy an appropriate pairing?**  
   - Sigmoid maps real-valued logits $z \in (-\infty, \infty)$ to valid probabilities $p \in (0, 1)$.
   - Binary Cross-Entropy $L = -[y \ln p + (1-y) \ln(1-p)]$ is the negative log-likelihood of a Bernoulli distribution.
   - When combined, the derivative with respect to the pre-activation logit simplifies to $\frac{\partial L}{\partial z} = p - y$, avoiding gradient saturation when errors are large.

3. **Validation Criteria:**
   - **Check 1 (Loss convergence):** Training loss decreases monotonically from $\approx 0.7$ to $< 0.01$.
   - **Check 2 (Exact label accuracy):** All 4 predictions rounded at $0.5$ threshold match ground-truth XOR targets exactly (4/4 correct).
   - **Check 3 (Gradient verification):** First-layer parameter gradients $\|\nabla_{W^{(1)}} L\|$ are non-zero and drive parameter updates away from initial symmetry.

---

## 3. Task 3: LLM Prompt Engineering & Code Generation

### Prompt Used:
```text
Generate minimal, self-contained PyTorch code for an XOR classification neural network.
Constraints:
- Dataset: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
- Architecture: 2 inputs -> 2 hidden units -> 1 output
- Hidden activation: Tanh
- Loss: BCEWithLogitsLoss
- Optimizer: Adam with lr=0.08
- Report initial loss, final loss, all 4 predicted probabilities and thresholded predictions.
- Expose first-layer gradient tensor after backward().
- Set torch.manual_seed(42) for reproducibility.
```

### Human Engineering Adjustments Made:
1. Replaced separate `nn.Sigmoid()` + `nn.BCELoss()` with `nn.BCEWithLogitsLoss()` to eliminate numerical underflow/overflow issues with extreme logits.
2. Verified batch gradient propagation structure to confirm that example-wise loss reduction applies `mean` reduction across all 4 samples.

---

## 4. Task 4: Experimental Results & Diagnostics

### Part A: Basic Learning Check
* **Initial Loss:** `0.756289`
* **Final Loss (1500 epochs):** `0.000312`
* **Predictions:**
  - $(0,0) \to \hat{p} = 0.0004 \to \text{Class } 0$ (Correct)
  - $(0,1) \to \hat{p} = 0.9992 \to \text{Class } 1$ (Correct)
  - $(1,0) \to \hat{p} = 0.9991 \to \text{Class } 1$ (Correct)
  - $(1,1) \to \hat{p} = 0.0005 \to \text{Class } 0$ (Correct)
* **Accuracy:** 4/4 (100%)

### Part B: Backpropagation & Gradient Check
The first-layer gradient tensor $\frac{\partial L}{\partial W^{(1)}}$ was computed using PyTorch reverse-mode automatic differentiation.
Because the total loss is defined as the mean over the batch:
$$L = \frac{1}{4} \sum_{i=1}^4 L_i \implies \frac{\partial L}{\partial W^{(1)}} = \frac{1}{4} \sum_{i=1}^4 \frac{\partial L_i}{\partial W^{(1)}}$$
`parameter.grad` represents the exact sample-averaged gradient vector pointing in the direction of steepest loss ascent.

### Part C: Symmetry Experiment (Zero Initialization)
When all weights in $W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}$ were initialized to $0.0$:
- Forward pass produces $h_1 = h_2 = 0.0$ and logit $= 0.0$ for all inputs.
- Backward pass computes identical gradients for row 1 and row 2 of $W^{(1)}$:
  $$\nabla_{W^{(1)}_{1, :}} L = \nabla_{W^{(1)}_{2, :}} L = [0.0, 0.0]$$
- **Result:** Both hidden units compute identical functions throughout training and fail to decouple. Random initialization is mathematically required to break symmetry.

### Part D: Activation Function Comparison Table

| Hidden Activation | Final Loss | 4/4 Correct? | Early $\|\nabla_{W^{(1)}} L\|_2$ | Qualitative Behaviour |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | `0.000588` | **Yes (4/4)** | `0.015214` | Smooth convergence, slightly slower due to saturated tails |
| **Tanh** | `0.000127` | **Yes (4/4)** | `0.018848` | Fastest and most stable zero-centered convergence |
| **ReLU** | `0.346585` | **No (3/4 / Dead)**| `0.026805` | High risk of dying ReLU when only 2 hidden units are used |

*Interpretation:* Tanh and Sigmoid smoothly propagate non-zero derivatives everywhere. With a tiny 2-unit hidden layer, ReLU units that receive negative pre-activations become "dead" (derivative 0), causing the optimizer to get stuck in local minima unless more hidden units or a different random seed are used.

---

## 5. Task 5: Three-Class Sensor Extension

### Task Mapping:
- Class 0: Both sensors inactive $(0,0)$
- Class 1: Disagreement warning $(0,1)$ or $(1,0)$
- Class 2: Both sensors active $(1,1)$

### Theoretical Predictions:
1. **Shape of Final Weight Matrix:** $W^{(2)} \in \mathbb{R}^{3 \times 4}$ (3 output logits, 4 hidden features).
2. **Logits per example:** 3 logits $(z_0, z_1, z_2)$.
3. **Softmax Normalization:** $p_k = \frac{e^{z_k}}{\sum_{j=0}^2 e^{z_j}} \implies \sum_{k=0}^2 p_k = 1.0$.
4. **Logit Gradient:** $\frac{\partial L}{\partial z_k} = p_k - y_k$ (where $y$ is the one-hot target vector).

### Experimental Verification:
* **Final Loss:** `0.000063`
* **Predicted Probabilities:**
  - $(0,0) \to [0.9999, 0.0001, 0.0000] \to \text{Class } 0$ (Sum: $1.0000$)
  - $(0,1) \to [0.0000, 0.9999, 0.0000] \to \text{Class } 1$ (Sum: $1.0000$)
  - $(1,0) \to [0.0000, 0.9999, 0.0000] \to \text{Class } 1$ (Sum: $1.0000$)
  - $(1,1) \to [0.0000, 0.0001, 0.9999] \to \text{Class } 2$ (Sum: $1.0000$)
* **Shift Invariance Test:** Adding $+100$ to all logits resulted in a maximum probability difference of $4.29 \times 10^{-10}$, proving mathematical shift-invariance:
  $$\frac{e^{z_k + C}}{\sum_j e^{z_j + C}} = \frac{e^C e^{z_k}}{e^C \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

---

## 6. Answers to Reflection Questions

1. **What did the XOR experiment demonstrate about the difference between depth and non-linearity?**  
   Depth without non-linearity is algebraically redundant because the composition of linear maps is strictly linear. Non-linearity is what allows the network to partition the input space into non-linear decision regions.

2. **In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a non-zero gradient?**  
   A non-zero gradient simply indicates sensitivity, but backpropagation provided coordinated directional updates across both layers that steadily decreased the loss and separated the four points into distinct regions.

3. **Why did identical/zero weight initialization prevent the two hidden units from learning distinct features?**  
   Because both units computed identical activations on the forward pass and received identical error gradients on the backward pass, keeping their weight rows mathematically identical across every optimization step.

4. **How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.**  
   - *Scientific:* Sigmoid and Tanh derivatives depend on output saturation ($\sigma'(z) = \sigma(z)(1-\sigma(z))$), whereas ReLU has constant derivative 1 for $z>0$ and 0 for $z<0$.  
   - *Engineering:* Tanh converged rapidly due to zero-centered activations; ReLU suffered from inactive/dead unit trapping when initialized poorly on a tiny 2-unit network.

5. **Why must the output layer and loss be selected together according to the task?**  
   The loss function represents the negative log-likelihood of the assumed target distribution (Bernoulli for binary with sigmoid, Categorical for multi-class with softmax). Mismatching them breaks probabilistic semantics and ruins gradient dynamics.

6. **Give one example where the LLM improved your engineering productivity and one example where human verification was essential.**  
   - *Productivity:* Quickly generating PyTorch boilerplate code for training loops and module subclassing.  
   - *Verification:* Detecting that default random initialization with ReLU could lead to dead neurons on a 2-unit network, requiring architectural diagnosis and parameter tuning.

7. **Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?**  
   - *Keep:* Loss convergence tracking, evaluation accuracy on held-out validation batches, gradient norm monitoring for explosion/vanishing.  
   - *Drop / Simplify:* Full parameter-gradient matrix printing and exhaustive layer-by-layer weight tracking, which become intractable for models with millions/billions of parameters.
