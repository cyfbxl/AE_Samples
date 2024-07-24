import torch
def fit(epoch, model, trainloader, testloader, opt):
    #训练
    correct = 0
    total = 0
    running_loss = 0
    model.train()  #训练模式
    for x, y in trainloader:
        result = model(x.cuda(), y.cuda())
        loss = model.loss_function(result)
        opt.zero_grad()
        loss['loss'].backward()
        opt.step()
        y_pred = torch.argmax(result[4], dim=1).cpu()
        correct += (y_pred == y).sum().item()
        total += y.size(0)
        running_loss += loss['loss'].item()
        
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total
        
    #测试    
    test_correct = 0
    test_total = 0
    test_running_loss = 0 
    model.eval()    #测试模式   
    for x, y in testloader:
        result = model(x.cuda(), y.cuda())
        loss = model.loss_function(result)
        y_pred = torch.argmax(result[4], dim=1).cpu()
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)
        test_running_loss += loss['loss'].item()
    
    epoch_test_loss = test_running_loss / len(testloader)
    epoch_test_acc = test_correct / test_total
    
        
    print('epoch: ', epoch+1, 
          'loss: ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
             )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc