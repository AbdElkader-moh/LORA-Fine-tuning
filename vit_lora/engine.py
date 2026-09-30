import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, desc="train"):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=desc)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(pixel_values=images).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
    return running_loss / len(loader.dataset)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1
