import numpy as np
import joblib
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
import torch
import random
from torchvision import datasets, transforms
import torchvision



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_minst_dataset(args):
    seed_everything(seed=args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32,antialias=True)
        # transforms.Resize(64,antialias=True)
        
    ])

    batch_size = 64
    train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_dl, test_dl


def get_cifar_dataset(args):
    seed_everything(seed=args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(32,antialias=True)       
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader

def get_dataset(args):
    print(113333)
    # sys.path.append('D:\桌面\pytest\VAE学习及示例')
    print(args.dataset_name)
    if args.dataset_name == 'minst':
        print(111)
        train_dl, test_dl = get_minst_dataset(args)
        return train_dl, test_dl
    elif args.dataset_name == 'cifar':
        train_dl, test_dl = get_cifar_dataset(args)
        return train_dl, test_dl
    
    
# -----------------1D
def data_prepare_XJTU(args,data_path = r'D:\桌面\XJTU_SY_dataset_4096.pkl'):
    seed_everything(seed=args.seed)
    all_pkl = joblib.load(data_path)
    rand_idx = np.random.permutation(10800)
    data_1 = np.vstack((all_pkl['train_1d'],all_pkl['test_1d']))[rand_idx]
    label_1 = np.concatenate((all_pkl['train_label'],all_pkl['test_label']))[rand_idx]

    train_ds = TensorDataset(torch.Tensor(data_1[:8000]), torch.Tensor(label_1[:8000]))
    test_ds = TensorDataset(torch.Tensor(data_1[-2000:]), torch.Tensor(label_1[-2000:]))
    train_dl = DataLoader(train_ds,args.batch_size,True,drop_last=True)
    test_dl = DataLoader(test_ds,args.batch_size,True,drop_last=True)
    
    return train_dl,test_dl


def data_prepare_SY(data_path = r'D:\桌面\pytest\仿真信号.pkl'):
    np.random.seed(21)
    simu = joblib.load(data_path)
    rand_idx = np.random.permutation(5000)
    data_1 = np.vstack((simu['train_1d'],simu['test_1d']))[rand_idx]
    label_1 = np.concatenate((simu['train_label'],simu['test_label']))[rand_idx]

    train_ds = TensorDataset(torch.Tensor(data_1[:4000]), torch.Tensor(label_1[:4000]))
    test_ds = TensorDataset(torch.Tensor(data_1[-1000:]), torch.Tensor(label_1[-1000:]))
    train_dl = DataLoader(train_ds,128,True,drop_last=True)
    test_dl = DataLoader(test_ds,128,True,drop_last=True)
    return train_dl,test_dl

#---------------------------------------------------------------------------------------------------------
normam_path = r'D:\博士\数据集整理\德国帕德博恩轴承数据集\pkl文件\N09_M07_F10_K006_normal_bearings.pkl'
outer_path = r'D:\博士\数据集整理\德国帕德博恩轴承数据集\pkl文件\N09_M07_F10_KA09_artificial_damage_OR.pkl'
iner_path = r'D:\博士\数据集整理\德国帕德博恩轴承数据集\pkl文件\N09_M07_F10_KI08_artificial_damage_IR.pkl'
def get_data(path, len_of_sample, num_of_sample):

    different_data = []
    different_label = []
    for this_path in path:
        this_data = joblib.load(this_path)
        print(path.index(this_path))
        
        for i in range(len(this_data)):
            for j in range(len(this_data[i]['data'])):
                # print(this_data[i]['working_condition'])
                middle_data = this_data[i]['data'][j]['data'][:len_of_sample * num_of_sample].reshape(num_of_sample, 1, len_of_sample)
                different_data.extend(middle_data)
                different_label.extend(np.zeros(middle_data.shape[0]) + path.index(this_path))
                
                

    different_data = torch.from_numpy(np.array(different_data)).type(torch.FloatTensor)   
    different_label = torch.from_numpy(np.array(different_label)).type(torch.FloatTensor)    
    return  different_data, different_label
        
class PU_dataset(torch.utils.data.Dataset):
    def __init__(self, normal_path, outer_path, iner_path, len_of_sample = 4096, num_of_sample = 50):
        self.path = [normal_path, outer_path, iner_path]
        
        self.species  = ['normal', 'outer', 'iner']
                        
        self.data, self.label = get_data(self.path, len_of_sample, num_of_sample)
        # seed_everything(seed=seed)
        random_array = np.random.permutation(len(self.label))
        self.data, self.label = self.data[random_array], self.label[random_array]

    def __getitem__(self, index):
        
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
def get_dataloader(dataset_of_PU, test_ratio = 0.2, batch_size = 120, is_shurrle = True):
    count = len(dataset_of_PU)
    train_count = int((1 - test_ratio)*count)
    test_count = count - train_count
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_of_PU, [train_count, test_count])
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = is_shurrle)
    test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle = is_shurrle)
    
    return train_dl, test_dl
def get_PU_dataset(seed):
    seed_everything(seed=seed)
    all_ds = PU_dataset(normam_path, outer_path, iner_path)
    train_dl, test_dl = get_dataloader(all_ds)
    return train_dl, test_dl
# a,b = next(iter(test_dl))
