"""
CS-F407: Artificial Intelligence
Laboratory - Neural Models: Learning, Depth, Activations, and Output Layers

Author: Arya Gupta
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def run_xor_binary():
    print("=" * 60)
    print("Task 4: Binary XOR Learning and Backpropagation Check")
    print("=" * 60)
    
    # XOR Dataset
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    torch.manual_seed(42)
    
    # 2-2-1 Model architecture
    class XORNet(nn.Module):
        def __init__(self, activation=nn.Tanh):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = activation()
            self.fc2 = nn.Linear(2, 1)
            
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))
            
    net = XORNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.08)
    
    init_logits = net(X)
    init_loss = criterion(init_logits, y).item()
    print(f"Initial Loss: {init_loss:.6f}")
    
    for epoch in range(1500):
        optimizer.zero_grad()
        logits = net(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    final_logits = net(X)
    probs = torch.sigmoid(final_logits).detach().numpy()
    preds = (probs > 0.5).astype(int)
    
    print(f"Final Loss: {final_loss:.6f}")
    print("Final Predicted Probabilities:")
    for i in range(4):
        print(f"  Input: {X[i].numpy()} -> Target: {int(y[i].item())} -> Prob: {probs[i][0]:.4f} -> Pred: {preds[i][0]}")
        
    print("\nFirst-layer Weight Gradient (dL/dW1):")
    print(net.fc1.weight.grad.numpy())


def run_symmetry_experiment():
    print("\n" + "=" * 60)
    print("Task 4 Part C: Symmetry Experiment (Zero Weight Initialization)")
    print("=" * 60)
    
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    net = nn.Sequential(
        nn.Linear(2, 2),
        nn.Tanh(),
        nn.Linear(2, 1)
    )
    
    # Initialize all weights and biases to 0.0
    for p in net.parameters():
        nn.init.constant_(p, 0.0)
        
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Tracking W1 weights and gradients over first 3 steps:")
    for step in range(3):
        optimizer.zero_grad()
        out = net(X)
        loss = criterion(out, y)
        loss.backward()
        print(f"Step {step}: Loss = {loss.item():.4f}")
        print(f"  W1 gradients:\n{net[0].weight.grad.numpy()}")
        optimizer.step()
        print(f"  W1 updated weights:\n{net[0].weight.data.numpy()}")
    print("Conclusion: Symmetrical initial weights yield identical gradients across hidden units, trapping hidden units in symmetry.")


def run_activation_comparison():
    print("\n" + "=" * 60)
    print("Task 4 Part D: Activation Function Comparison")
    print("=" * 60)
    
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    
    activations = {
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'ReLU': nn.ReLU
    }
    
    results = []
    criterion = nn.BCEWithLogitsLoss()
    
    for name, act_cls in activations.items():
        torch.manual_seed(100)
        net = nn.Sequential(
            nn.Linear(2, 4),
            act_cls(),
            nn.Linear(4, 1)
        )
        optimizer = optim.Adam(net.parameters(), lr=0.05)
        
        # Measure early gradient norm (step 0)
        optimizer.zero_grad()
        out = net(X)
        loss = criterion(out, y)
        loss.backward()
        early_grad_norm = torch.norm(net[0].weight.grad).item()
        
        # Train
        for step in range(2000):
            optimizer.zero_grad()
            out = net(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        final_loss = loss.item()
        probs = torch.sigmoid(net(X)).detach().numpy()
        preds = (probs > 0.5).astype(int)
        correct = (preds.flatten() == y.numpy().flatten()).all()
        
        results.append((name, final_loss, bool(correct), early_grad_norm, preds.flatten()))
        
    print(f"{'Hidden Activation':<18} | {'Final Loss':<12} | {'4/4 Correct?':<12} | {'Early ||grad_W1||':<15}")
    print("-" * 65)
    for name, fl, corr, gn, _ in results:
        print(f"{name:<18} | {fl:<12.6f} | {str(corr):<12} | {gn:<15.6f}")


def run_three_class_extension():
    print("\n" + "=" * 60)
    print("Task 5: Three-Class Sensor Decision Extension")
    print("=" * 60)
    
    # Class 0: (0,0) [Both inactive]
    # Class 1: (0,1), (1,0) [Sensors disagree]
    # Class 2: (1,1) [Both active]
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_multi = torch.tensor([0, 1, 1, 2])
    
    torch.manual_seed(42)
    net_multi = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 3)
    )
    
    optimizer = optim.Adam(net_multi.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1500):
        optimizer.zero_grad()
        logits = net_multi(X)
        loss = criterion(logits, y_multi)
        loss.backward()
        optimizer.step()
        
    logits = net_multi(X)
    probs = torch.softmax(logits, dim=-1).detach().numpy()
    preds = probs.argmax(axis=1)
    
    print(f"Final 3-Class Cross-Entropy Loss: {loss.item():.6f}")
    print("Predicted Softmax Probabilities:")
    for i in range(4):
        print(f"  Input: {X[i].numpy()} -> True Class: {y_multi[i]} -> Probs: {probs[i]} -> Pred: {preds[i]} (Sum: {probs[i].sum():.4f})")
        
    # Check Softmax Shift Invariance (+100)
    shifted_logits = logits + 100.0
    shifted_probs = torch.softmax(shifted_logits, dim=-1).detach().numpy()
    max_diff = np.abs(probs - shifted_probs).max()
    print(f"Softmax Shift Invariance Check (+100 max diff): {max_diff:.2e}")


if __name__ == '__main__':
    run_xor_binary()
    run_symmetry_experiment()
    run_activation_comparison()
    run_three_class_extension()
