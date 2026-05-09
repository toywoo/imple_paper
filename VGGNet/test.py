import torch

def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # TenCrop 데이터(5차원)인지 확인
            if X.dim() == 5:
                # X shape: [batch, 10, channel, height, width]
                bs, ncrops, c, h, w = X.size()
                # 10개의 이미지를 하나의 배치처럼 합쳐서 모델 통과
                result = model(X.view(-1, c, h, w))
                # 결과값을 다시 [batch, 10, classes]로 돌린 뒤 평균(mean) 계산
                pred = result.view(bs, ncrops, -1).mean(1)
            else:
                pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")