
def train_AE(model, optimizer, train_dl, test_dl, criterion, n_epochs):
    model.train()
    for epoch in range(1, n_epochs+1):
    # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_dl:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            images = images.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_dl)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
    
import torch    
def fit(epoch, model, trainloader, testloader, opt, loss_fn):
    #训练
    correct = 0
    total = 0
    running_loss = 0
    model.train()  #训练模式
    for x, y in trainloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        y_pred = torch.argmax(y_pred, dim=1)
        correct += (y_pred == y).sum().item()
        total += y.size(0)
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total
        
    #测试    
    test_correct = 0
    test_total = 0
    test_running_loss = 0 
    model.eval()    #测试模式   
    for x, y in testloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)
        test_running_loss += loss.item()
    
    epoch_test_loss = test_running_loss / len(testloader)
    epoch_test_acc = test_correct / test_total
    
        
    print('epoch: ', epoch+1, 
          'loss: ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
             )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
