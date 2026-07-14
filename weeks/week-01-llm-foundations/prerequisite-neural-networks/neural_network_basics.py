import torch
import torch.nn as nn
import torch.optim as optim

print("=== Day 0: Understanding Neural Networks ===\n")

# =============================================
# 1. Simple Neural Network (1 Layer)
# =============================================

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear layer: input -> output
        self.linear = nn.Linear(in_features=2, out_features=1)  # 2 inputs, 1 output
    
    def forward(self, x):
        return self.linear(x)

# Create the model
model = SimpleNN()

print("Simple Neural Network created!")
print("It has one layer that takes 2 inputs and gives 1 output.\n")

# Example input
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 samples, each with 2 features
print("Input data:")
print(x)

# Forward pass
output = model(x)
print("\nOutput (prediction):")
print(output)

# =============================================
# 2. Training Loop Example (Simple)
# =============================================

print("\n" + "="*50)
print("Simple Training Example (Learning)")
print("="*50)

# Let's pretend we want the model to predict 5.0 for input [2.0, 3.0]
target = torch.tensor([[5.0]])

loss_fn = nn.MSELoss()           # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Simple optimizer

for epoch in range(10):  # Train for 10 steps
    # Forward pass
    prediction = model(x)
    loss = loss_fn(prediction[0], target)  # Compare first prediction with target
    
    # Backward pass + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Prediction: {prediction[0].item():.4f}")

print("\nTraining finished!")
print("The model has learned to adjust its weights to reduce error.")

# =============================================
# Key Concepts Summary
# =============================================
print("\n" + "="*50)
print("Key Concepts You Learned Today:")
print("="*50)
print("• Neural Network = Layers of neurons connected together")
print("• Forward Pass = Input → Prediction")
print("• Loss = How wrong the prediction was")
print("• Backward Pass = Calculate gradients (how to improve)")
print("• Optimizer = Updates weights to reduce loss")
print("• Training = Repeat forward + backward many times")