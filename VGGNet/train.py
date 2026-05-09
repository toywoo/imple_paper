import torch
import time

def train(device, dataloader, num_epochs, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        running_loss = 0.0
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 예측 및 손실 계산
            pred = model(inputs)
            loss = loss_fn(pred, labels)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            loss_history.append(loss_val)

            if batch % 10 == 0:
                current = batch * len(inputs)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds\n")

    return loss_history
