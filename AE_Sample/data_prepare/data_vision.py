import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
def pre_visualization(train_dl, test_dl, args):
    batch_size_to_show = args.batch_size

    data_iter = next(iter(test_dl))
    images, labels = data_iter
    print(images.shape)
    # 顯示圖片
    def imshow(img):
        img = img.permute(1, 2, 0)
        plt.imshow(img, cmap='gray')
        plt.show()

    # 顯示數據集中的圖片和標籤
    imshow(torchvision.utils.make_grid(images))
    print('标签: ', ' '.join(str(labels[j].item()) for j in range(batch_size_to_show)))

def minst_after_train_see(model, test_dl, args):
    # obtain one batch of test images
    images, labels = next(iter(test_dl))

    # get sample outputs
    output = model(images.cuda()).cpu()
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of iages
    output = output.view(args.batch_size, 1, 32, 32)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(12,2))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            
def cifar_after_train_see(model, test_dl, args):
    # obtain one batch of test images
    images, labels = next(iter(test_dl))

    # get sample outputs
    output = model(images.cuda()).cpu()
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of iages
    output = output.view(args.batch_size, 3, 32, 32)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(12,2))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            # ax.imshow(np.squeeze(img), cmap='gray')
            ax.imshow(img.transpose(1, 2, 0))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            
            
def pre_1d_see(train_dl, test_dl, args):
    x,y = next(iter(train_dl))
    plt.figure(figsize=(12,6))
    for i in range(args.num_of_see):
        ax = plt.subplot(args.num_of_see, 1, i + 1)
        ax.plot(x[i][0])
        ax.set_title(f'Original Signal {i+1}')
        ax.set_xlim(0,4096)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        # ax.set_xlabel('Sample')
        plt.ylabel('Amplitude')
    plt.tight_layout()    
    print(y[0:i+1])
    
def after_1d_see(model, test_dl, args):
    x,y = next(iter(test_dl))
    
    plt.figure(figsize=(12,7))
    x_hat = model(x.cuda()).cpu().detach().numpy()

    for i in range(args.num_of_see):
        # 原始信号
        ax1 = plt.subplot(args.num_of_see, 2, 2*i+1)
        ax1.plot(x[i][0])
        ax1.set_title(f'Original Signal {i+1}')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        # 仅显示 x 和 y 轴
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 重构信号
        ax2 = plt.subplot(args.num_of_see, 2, 2*i+2)
        ax2.plot(x_hat[i][0])
        ax2.set_title(f'Reconstructed Signal {i+1}')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')
        # 仅显示 x 和 y 轴
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        

    plt.tight_layout()
    plt.show()

def minst_VAE_after_train_see(model, test_dl, args):
    images, labels = next(iter(test_dl))

    # get sample outputs
    output, pre_label = model(images.cuda())[0].cpu(), model(images.cuda())[-1].cpu()
    pre_label = torch.argmax(pre_label, dim=1)[:10]
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of iages
    output = output.view(args.batch_size, 1, 32, 32)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(12,2))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            # ax.imshow(img.transpose(1, 2, 0))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    print(pre_label)
    
    
def VAE_1d_after_train_see(model, test_dl, args):
    # def after_1d_see(model, test_dl, args):
    x,y = next(iter(test_dl))
    
    plt.figure(figsize=(12,7))
    x_hat = model(x.cuda())[0].cpu().detach().numpy()

    for i in range(args.num_of_see):
        # 原始信号
        ax1 = plt.subplot(args.num_of_see, 2, 2*i+1)
        ax1.plot(x[i][0])
        ax1.set_title(f'Original Signal {i+1}')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        # 仅显示 x 和 y 轴
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 重构信号
        ax2 = plt.subplot(args.num_of_see, 2, 2*i+2)
        ax2.plot(x_hat[i][0])
        ax2.set_title(f'Reconstructed Signal {i+1}')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')
        # 仅显示 x 和 y 轴
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        

    plt.tight_layout()
    plt.show()
