import os
import re
import sys
from pathlib import Path

import numpy as np
import random
import torch
from torch.utils.data import DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   
    random.seed(seed)
    np.random.seed(seed)

def clear_models(s):
    path_re = s + "*"
    for filename in os.listdir("model/"):
        if re.search(path_re, filename):
            os.system("rm model/{}*".format(s))
            break

def sep_print(str):
    '''Separate the print information'''
    print()
    print("-------------------------------", end='  ')
    print(str,end='  ')
    print("-------------------------------")
    
def check_print(x, s="Unspecified"):
    '''Print the value and type de variable x whose name is s'''
    sep_print("Checking  --->  " + s)
    print("Value:", x)
    print("Type:", type(x))
    if torch.is_tensor(x):
        print("Shape: ", x.shape)
        if len(x.shape) == 0:
            print("0-Dimension: ", x.item())
        else:
            print("Sample: ", x[0])
    sep_print("Finish Checking  ---->  " + s)
    
def path_control():
    '''Import path directory'''
    sep_print("Start")
    base_path = str(Path(__file__).resolve().parent.parent)
    sys.path.append(base_path)
    sep_print("Project Path")
    print(base_path)

def get_device():
    '''determine the supported device'''
    sep_print("Device")
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        print("Using GPU\n")
    else:
        device = torch.device('cpu') # don't have GPU 
        print("Using CPU")
    return device

def load_split_data(dataset, ratio, batch_size, s="Unknown Dataset"):
    '''Prepare the dataset according to the train/test ratio and also the batch size '''
    # sep_print("{} - Data Load & Split".format(s))
    N = len(dataset)
    train_size = int(ratio * N)
    test_size = N - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(test_dataset) > 0:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_dataloader = None
    # features, label = next(iter(train_dataloader))
    # print("Train size: {}; Test size: {}".format(len(train_dataset), len(test_dataset)))
    # print("Feature shape: ", features.shape)
    # print("label shape: ", label.shape)
    return train_dataloader, test_dataloader

class ReplayBuffer(object):
    def __init__(self, 
                 state_len, state_dim, action_dim, rep_dim,
                 max_size, 
                 device):
        # Size control
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Experience
        self.state = np.zeros((max_size, state_len, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.epis_rep = np.zeros((max_size, rep_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_len, state_dim))
        self.done = np.zeros((max_size, 1))
        
        # Device
        self.device = device

    def add(self, state, action, epis_rep, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.epis_rep[self.ptr] = epis_rep
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        # Size update
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.epis_rep[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
