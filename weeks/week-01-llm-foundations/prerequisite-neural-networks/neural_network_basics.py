import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("=== Prerequisite — Neural Networks Fundamentals ===\n")

# =============================================
# Simple Neural Network
# =============================================
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleNN()

# =============================================
# TRAINING DATA (What we use to teach the model)
# =============================================
print("Training Examples (Used for Learning):")
training_data = [
    ([1., 2.], 3.),
    ([3., 4.], 7.),
    ([5., 1.], 6.),
    ([2., 8.], 10.),
    ([4., 5.], 9.)
]

for x_data, y_true in training_data:
    print(f"   Input: {x_data} → Target Output: {y_true}")

print("\nNote: We never showed the model [2, 3] during training.\n")

# =============================================
# Training Loop
# =============================================
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []

print("Training the model...\n")

for epoch in range(50):
    total_loss = 0
    for x_data, y_true in training_data:
        x = torch.tensor([x_data])
        target = torch.tensor([[y_true]])
        
        prediction = model(x)
        loss = loss_fn(prediction, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(training_data)
    losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.6f}")

print("\nTraining Finished!")

# =============================================
# TEST EXAMPLE (Unseen data - Checking if model learned)
# =============================================
print("\n" + "="*60)
print("TEST EXAMPLE (Never seen during training)")
print("="*60)

test_input = torch.tensor([[2., 3.]])   # This was NOT in training data
prediction = model(test_input)

print(f"Input  : [2.0, 3.0]")
print(f"Target : 5.0")
print(f"Model Predicted: {prediction.item():.4f}")
print(f"Error: {abs(prediction.item() - 5.0):.4f}")

# =============================================
# Loss Curve Visualization
# =============================================
plt.figure(figsize=(8, 5))
plt.plot(losses, marker='o', color='blue')
plt.title("Neural Network Learning - Loss Curve")
plt.xlabel("Epoch (Training Steps)")
plt.ylabel("Loss (How Wrong the Model Was)")
plt.grid(True)
plt.show()