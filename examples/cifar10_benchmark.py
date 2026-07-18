#!/usr/bin/env python
"""Benchmark a standard CNN against a potential CNN on CIFAR-10."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim

from quantum_plumbing import PotentialActivation, PotentialConv2d, PotentialSequential, potential_loss

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install torchvision to run this benchmark: pip install torchvision") from exc


class PotentialConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = PotentialSequential(
            PotentialConv2d(3, 16, kernel_size=3, padding=1, num_potentials=4),
            PotentialActivation("relu"),
            PotentialConv2d(16, 32, kernel_size=3, padding=1, num_potentials=4),
            PotentialActivation("relu"),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor):
        x, H = self.features(x)
        x = self.pool(x).flatten(1)
        logits = self.classifier(x)
        return logits, H


def make_loaders(batch_size: int = 128):
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
    )


def train_model(model, train_loader, device: torch.device, epochs: int = 2, potential: bool = False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if potential:
                logits, H = model(x)
                loss = potential_loss(logits, y, H=H, h_diversity_weight=0.01)
            else:
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def evaluate(model, test_loader, device: torch.device, potential: bool = False) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)[0] if potential else model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders()
    baseline = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10),
    ).to(device)
    potential = PotentialConvClassifier().to(device)
    baseline_time = train_model(baseline, train_loader, device)
    potential_time = train_model(potential, train_loader, device, potential=True)
    print({"baseline_accuracy": evaluate(baseline, test_loader, device), "baseline_train_time_sec": baseline_time})
    print({"potential_accuracy": evaluate(potential, test_loader, device, potential=True), "potential_train_time_sec": potential_time})


if __name__ == "__main__":
    main()
