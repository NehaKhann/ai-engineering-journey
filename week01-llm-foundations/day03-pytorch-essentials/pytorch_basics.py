import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =============================================
# Day 3: PyTorch Essentials - With Loss Curve
# =============================================

torch.manual_seed(42)

print("=== PyTorch Basics - Training a Tiny Model ===\n")

# Data
x = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([10.0])

# Model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=3, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

model = TinyModel()
loss_fn = nn.MSELoss()

# Training settings
learning_rate = 0.01
epochs = 20
losses = []   # To store loss values for plotting

print("Starting Training...\n")

for epoch in range(epochs):
    # Forward Pass
    prediction = model(x)
    loss = loss_fn(prediction, target)
    
    # Record loss for plotting
    losses.append(loss.item())
    
    # Backward Pass
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        model.linear.weight -= learning_rate * model.linear.weight.grad
        model.linear.bias -= learning_rate * model.linear.bias.grad
        
        # Clear gradients
        model.linear.weight.grad.zero_()
        model.linear.bias.grad.zero_()
    
    # Print progress
    if epoch % 2 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Prediction: {prediction.item():.4f}")

print("\n=== Training Finished ===")
print(f"Final Prediction: {model(x).item():.4f} (Target was 10.0)")

# =============================================
# Plot the Loss Curve
# =============================================
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', linestyle='-', color='b')
    plt.title("Loss Curve - How the Model Improved Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
except ImportError:
    print("\nMatplotlib is not installed. Install it using:")
    print("pip install matplotlib")