#!/usr/bin/env python
"""Check whether H-diversity regularization helps low-data MNIST generalization."""

from __future__ import annotations

import random

import torch
import torch.optim as optim

from quantum_plumbing import PotentialMLP, potential_loss

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install torchvision to run this benchmark: pip install torchvision") from exc


def subset_dataset(dataset, fraction: float = 0.1):
    count = int(len(dataset) * fraction)
    indices = random.sample(range(len(dataset)), count)
    return torch.utils.data.Subset(dataset, indices)


def make_loaders(batch_size: int = 128, fraction: float = 0.1):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return (
        torch.utils.data.DataLoader(subset_dataset(train, fraction=fraction), batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
    )


def train_and_eval(diversity_weight: float) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders()
    model = PotentialMLP([28 * 28, 128, 10], num_potentials=4, dropout_p=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, H = model(x)
            loss = potential_loss(logits, y, H=H, h_diversity_weight=diversity_weight)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    without_diversity = train_and_eval(0.0)
    with_diversity = train_and_eval(0.01)
    print({"without_diversity": without_diversity, "with_diversity": with_diversity})


if __name__ == "__main__":
    main()
