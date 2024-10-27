import argparse
import torch
'''这是说明和设置联邦学习参数的文件'''


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--GE",  type=int, default=50, help='number of Global round of training')
    parser.add_argument("--LE", type=int, default=100, help='number of Local round of training')
    parser.add_argument("--lr", type=int, default=0.001, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0, help='learning rate decay per global round')
    parser.add_argument("--K", type=int, default=5, help='number of total client')
    parser.add_argument("--C", type=float, default=1, help='sampling rate')
    parser.add_argument("--input_dim", type=int, default=8, help='input dimension')
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--optimizer", type=str, default='adam', help='type of optimizer')
    parser.add_argument("--mu",  type=float, default=0.0001, help='FedProx parameter mu')
    parser.add_argument("--importance", type=float, default=1000, help='Heyper Parameters of EWC')
    parser.add_argument("--alphaK", type=int, default=1, help='lambda/sqrt(GLOABL-ITRATION) according to the paper')
    parser.add_argument("--lamda", type=int, default=1, help='Regularization weight')
    parser.add_argument("--sigma", type=int, default=1, help='FedAMP Heyper Parameters')
    clients = ['server' + str(i) for i in range(0, 10)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()
    return args
