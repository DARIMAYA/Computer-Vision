# train_classifier.py
import torch
import numpy as np
import os
from detection_and_metrics import fit_cls_model

# Correct path to training data
data_path = 'tests/00_unittest_classifier_input/train_data.npz'

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found!")
    exit(1)

print(f"Loading training data from: {data_path}")

# Load training data
data = np.load(data_path)
X = data['X']
y = data['y']

print(f"Loaded {len(X)} training samples")
print(f"Original X shape: {X.shape}, y shape: {y.shape}")

# Fix data format: convert to PyTorch format [batch, channels, height, width]
if X.ndim == 3:
    # If shape is (N, H, W), add channel dimension
    X = X[:, np.newaxis, :, :]
    print(f"Added channel dimension")
elif X.ndim == 4:
    # Check if shape is (N, H, W, C) and convert to (N, C, H, W)
    if X.shape[-1] in [1, 3]:  # Last dimension is channels
        print(f"Converting from (N, H, W, C) to (N, C, H, W)")
        X = np.transpose(X, (0, 3, 1, 2))
    elif X.shape[1] not in [1, 3]:
        # Probably (N, H, W, C) format where C is not obvious
        print(f"Assuming (N, H, W, C) format, converting to (N, C, H, W)")
        X = np.transpose(X, (0, 3, 1, 2))

# If RGB, convert to grayscale
if X.shape[1] == 3:
    print("Converting RGB to grayscale")
    X = 0.299 * X[:, 0:1, :, :] + 0.587 * X[:, 1:2, :, :] + 0.114 * X[:, 2:3, :, :]

print(f"Processed X shape: {X.shape}")
print(f"Expected shape: (N, 1, 40, 100)")

# Verify shape
assert X.shape[1] == 1, f"Expected 1 channel, got {X.shape[1]}"
assert X.shape[2] == 40, f"Expected height 40, got {X.shape[2]}"
assert X.shape[3] == 100, f"Expected width 100, got {X.shape[3]}"

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

print(f"\nClass distribution:")
print(f"  Class 0 (background): {(y == 0).sum().item()}")
print(f"  Class 1 (car): {(y == 1).sum().item()}")

# Train model
print("\nTraining classifier (this may take a while)...")
model = fit_cls_model(X, y, fast_train=False)

# Save model
torch.save(model.state_dict(), 'classifier_model.pt')
print("\nModel saved to classifier_model.pt")

# Test accuracy
model.eval()
with torch.no_grad():
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).float().mean()

    # Per-class accuracy
    class0_mask = y == 0
    class1_mask = y == 1
    class0_acc = (predicted[class0_mask] == y[class0_mask]).float().mean()
    class1_acc = (predicted[class1_mask] == y[class1_mask]).float().mean()

    print(f"\nTraining accuracy: {accuracy:.4f}")
    print(f"  Class 0 (background) accuracy: {class0_acc:.4f}")
    print(f"  Class 1 (car) accuracy: {class1_acc:.4f}")