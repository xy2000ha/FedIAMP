import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
import os


def my_save_model(net):
    net_dir = './saved_models'
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
    net_path = os.path.join(net_dir, 'carbon.pth')
    torch.save(net, net_path)


def get_data():
    # 得到原始数据集，格式为数组
    path = './DATA/data.xlsx'
    data = pd.read_excel(io=path, sheet_name='DATA1', usecols='A:I')
    data = np.asarray(data.astype(np.float32))
    return data


def data_process(data):
    # 数据处理，将原始数据标准化，返回值有处理后的数据,归一化模型
    Scale = MinMaxScaler()
    data = Scale.fit_transform(data)
    data = torch.from_numpy(data)
    return data, Scale


def spilt_feature_Label(data):
    """ spilt origin dataset into feature and label"""
    a, b = data.shape
    train = data[:, 0:b - 1]
    label = data[:, b - 1]
    # print("train:", train)
    # print("label", label)
    return train, label


def convert_to_tensor(data):
    """将nparray转变成tensor"""
    if isinstance(data, np.ndarray):
        output = torch.from_numpy(data.astype(np.float32))
    elif isinstance(data, torch.Tensor):
        output = data
    else:
        raise TypeError("Input data type not supported")
    return output


def create_set():
    """randomly create dataset """

    return train_set, test


def create_set1():
    """create dataset"""
   
    return train_set, test
