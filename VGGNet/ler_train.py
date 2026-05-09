import torch
import time

def train_one_epoch(device, dataloader, model, loss_fn, optimizer, epoch_idx, loss_history):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0

    print(f"Epoch {epoch_idx}\n-------------------------------")
    start_time = time.time()
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        pred = model(inputs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val) # 모든 배치 손실 기록
        running_loss += loss_val

        if batch % 10 == 0:
            current = batch * len(inputs)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
            
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch_idx} finished in {epoch_duration:.2f} seconds\n")

    return running_loss / len(dataloader)
