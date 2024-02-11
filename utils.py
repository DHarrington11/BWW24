import torch
import torch.nn.functional as nnf


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
    print(f"Loss: {loss}")


def test(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred.squeeze(), y)
            total_loss += loss
    print(f"Test Loss: {total_loss}\n")
