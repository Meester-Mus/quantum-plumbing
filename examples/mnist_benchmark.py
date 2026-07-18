#!/usr/bin/env python
"""Benchmark PotentialMLP against a standard MLP on MNIST."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim

from quantum_plumbing import PotentialMLP, h_utilization, potential_loss

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install torchvision to run this benchmark: pip install torchvision") from exc


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def make_loaders(batch_size: int = 128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
    )


def train_standard(train_loader, test_loader, device: torch.device, epochs: int = 3):
    model = nn.Sequential(nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total, "train_time_sec": elapsed}


def train_potential(train_loader, test_loader, device: torch.device, epochs: int = 3):
    model = PotentialMLP([28 * 28, 128, 10], num_potentials=4, dropout_p=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    epoch_utils = []
    for _ in range(epochs):
        model.train()
        batch_utils = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, H = model(x)
            loss = potential_loss(logits, y, H=H, h_diversity_weight=0.01)
            loss.backward()
            optimizer.step()
            batch_utils.append(h_utilization(H).item())
        epoch_utils.append(sum(batch_utils) / len(batch_utils))
    elapsed = time.perf_counter() - started

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return {
        "accuracy": correct / total,
        "train_time_sec": elapsed,
        "avg_h_utilization": sum(epoch_utils) / len(epoch_utils),
        "epoch_h_utilization": epoch_utils,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders()
    baseline = train_standard(train_loader, test_loader, device)
    potential = train_potential(train_loader, test_loader, device)
    print("Standard MLP:", baseline)
    print("Potential MLP:", potential)


if __name__ == "__main__":
    main()
