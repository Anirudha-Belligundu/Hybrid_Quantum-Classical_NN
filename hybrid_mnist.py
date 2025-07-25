#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Quantum–Classical MNIST classifier

Tested with:
  • Python 3.9
  • torch 2.2.2 (CUDA 12.1 build) / CPU fallback
  • torchquantum 0.1.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import PauliZ                   # NEW: for measurement
import numpy as np

import matplotlib
matplotlib.use("Agg")                             # headless back‑end
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class QuantumModel(tq.QuantumModule):
    def __init__(self, n_qubits: int = 4, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Parameterised single‑qubit rotations and CZ entanglers
        self.ry_layers = tq.QuantumModuleList(
            [tq.Op1QAllLayer(op=tq.RY, n_wires=n_qubits, trainable=True)
             for _ in range(n_layers)]
        )
        self.entangle_layers = tq.QuantumModuleList(
            [tq.Op2QAllLayer(op=tq.CZ, n_wires=n_qubits, circular=True)
             for _ in range(n_layers)]
        )

        # NEW: built‑in measurement layer (⟨Z⟩ on every qubit)
        self.measure = tq.MeasureAll(PauliZ)

    def forward(self, q_dev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        # ----- Data encoding ------------------------------------------------ #
        for i in range(self.n_qubits):
            tqf.ry(q_dev, wires=i, params=x[:, i])

        # ----- Variational layers ------------------------------------------ #
        for layer in range(self.n_layers):
            self.entangle_layers[layer](q_dev)
            self.ry_layers[layer](q_dev)

        # ----- Measurement (returns shape: [batch, n_qubits]) -------------- #
        return self.measure(q_dev)



class HybridModel(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(14 * 14, 64)
        self.fc2 = nn.Linear(64, n_qubits)

        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.quantum_circuit = QuantumModel(n_qubits, n_layers)

        self.fc3 = nn.Linear(n_qubits, 10)     # 10 MNIST classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Classical preprocessing ------------------------------------- #
        x = x.view(-1, 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x) * np.pi           # → [0, π]

        # ----- Quantum processing ------------------------------------------ #
        self.q_device.reset_states(bsz=x.size(0))
        x = self.quantum_circuit(self.q_device, x)

        # ----- Classical post‑processing ----------------------------------- #
        x = self.fc3(x)
        return x



def get_data_loaders(batch_size: int = 32):
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ])
    train_data = datasets.MNIST("./data",
                                train=True,
                                download=True,
                                transform=transform)
    test_data = datasets.MNIST("./data",
                               train=False,
                               transform=transform)
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size))



def train_model(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx:4}/{len(train_loader)}] "
                  f"loss = {loss.item():.4f}")
    print(f"Epoch {epoch} completed.")


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"\nTest accuracy: {acc:.2f}%")

    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n",
          classification_report(all_targets, all_preds, digits=4))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Confusion matrix saved → confusion_matrix.png")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_data_loaders(batch_size=32)
    model = HybridModel(n_qubits=4, n_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training …")
    for epoch in range(1, 6):                  # train 5 epochs
        train_model(model, train_loader, optimizer, epoch, device)

    print("\nEvaluating …")
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()

