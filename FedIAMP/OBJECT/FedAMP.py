import copy
import numpy as np
import torch
from PREPROCESS import args
from MY_MODEL.mlp import ANN
from PREPROCESS.preprocess import *
from TRIAIN_FUNC.client_train import FedAP_train, test
import math


class FedAMP:
    def __init__(self, args):
        self.args = args
        self.nn = ANN(args=self.args, name='server')
        self.nns = []
        self.train_set, self.test_set = create_set1()
        self.Scale = data_process(get_data())[1]
        self.alphaK = args.alphaK
        self.lamda = args.lamda
        self.sigma = args.sigma
        self.G_model = copy.deepcopy(self.nn)
        self.temp_client = copy.deepcopy(self.nn)

        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

        self.temp_server = copy.deepcopy(self.nns)

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()


    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def e(self, x):
        return math.exp(-x / self.sigma) / self.sigma

    def send_model(self, index):
        # self.temp_server = copy.deepcopy(self.nns)
        for i in index:
            c = copy.deepcopy(self.nns[i])
            mu = copy.deepcopy(self.G_model)
            for param in mu.parameters():
                param.data.zero_()
            coef = torch.zeros(self.args.K)
            for j in index:
                if i != j:
                    mw = copy.deepcopy(self.nns[j])
                    weights_i = torch.cat([p.data.view(-1) for p in c.parameters()], dim=0)
                    weights_j = torch.cat([p.data.view(-1) for p in mw.parameters()], dim=0)
                    sub = (weights_i - weights_j).view(-1)
                    sub = torch.dot(sub, sub)
                    coef[j] = self.alphaK * self.e(sub)
                else:
                    coef[j] = 0
            coef_self = 1 - torch.sum(coef)
            # print(i, coef)
            for j in index:
                mw = self.G_model
                for param, param_j in zip(mu.parameters(), mw.parameters()):
                    param.data += coef[j] * param_j
            self.temp_client = copy.deepcopy(self.G_model)
            for param in self.temp_client.parameters():
                param.data.zero_()
            for new_param, old_param, self_param in zip(mu.parameters(),
                                                        self.temp_client.parameters(), self.nns[i].parameters()):
                old_param.data = (new_param.data + coef_self * self_param.data).clone()
            self.temp_server[i] = copy.deepcopy(self.temp_client)

    def servers(self):
        index2 = np.linspace(0, self.args.K - 1, self.args.K)
        index2 = index2.astype(int)
        # self.dispatch(index2)
        for t in range(self.args.GE):
            print('server round epoch:', t + 1, ':')
            self.send_model(index2)
            for j in index2:
                self.temp_client = copy.deepcopy(FedAP_train(self.args, self.nns[j], self.nn, self.train_set[j]))
                self.temp_client.len = len(self.train_set[j])
                self.temp_server[j] = copy.deepcopy(self.temp_client)
            self.nns = copy.deepcopy(self.temp_server)
            # self.local_test()
            self.aggregation(index2)
            self.G_model = copy.deepcopy(self.nn)
        # self.dispatch(index2)

    def local_test(self):
        for i in range(len(self.nns)):
            model = self.nns[i]
            model.eval()
            for client in self.args.clients:
                model.name = client
            print("test:", i)
            test(self.args, self.test_set, self.Scale, model)
            # if i == 4:
            #     # 根据需要保存模型，这里将目标设为4了，优化需要更改
            #     my_save_model(self.nns[i])

    def global_test(self):
        model = self.nn
        model.eval()
        model.name = self.args.clients[1]
        test(self.args, self.test_set, self.Scale, model)


# ，一个是server，send_model()里的index个ceof计算在检查一下
    # def my_server(self):
    #     index2 = np.linspace(0, self.args.K - 1, self.args.K)
    #     index2 = index2.astype(int)
    #     index = self.get_rand_index()
    #     torch.set_printoptions(precision=10)
    #     for t in range(self.args.GE):
    #         for j in index2:
    #             self.temp_client = copy.deepcopy(FedAP_train(self.args, self.nns[j], self.nn, self.train_set[j]))
    #             self.temp_server[j] = copy.deepcopy(self.temp_client)
    #             self.nns = copy.deepcopy(self.temp_server)
    #             self.nns[j].len = len(self.train_set[j])
    #             print(j, self.nns[j].len)
    #         self.aggregation(index)
    #         self.G_model = copy.deepcopy(self.nn)
    #         # for j in index2:
    #         #     print("parameters nns:", j)
    #         #     for p in self.nns[j].parameters():
    #         #         print(p)
    #         coef = torch.zeros(self.args.K)
    #         for i in index:
    #             c = copy.deepcopy(self.nns[i])
    #             for j in index:
    #                 mw = copy.deepcopy(self.nns[j])
    #                 if i != j:
    #                     weights_i = torch.cat([p.data.view(-1) for p in c.parameters()], dim=0)
    #                     weights_j = torch.cat([p.data.view(-1) for p in mw.parameters()], dim=0)
    #                     sub = (weights_i - weights_j).view(-1)
    #                     # print("sub:", sub)
    #                     sub = torch.dot(sub, sub)
    #                     print("sub^2:", sub)
    #                     coef[j] = self.alphaK * self.e(sub)
    #                 else:
    #                     coef[j] = 0
    #                 coef_self = 1 - torch.sum(coef)
    #             print(i, coef)
    #         self.send_model(index)


# class FedAMP:
#     def __init__(self, args):
#         self.args = args
#         self.nn = ANN(args=self.args, name='server')
#         self.nns = []
#         self.train_set, self.test_set = create_set2()
#         self.Scale = data_process(get_data())[1]
#         self.alphaK = args.alphaK
#         self.lamda = args.lamda
#         self.sigma = args.sigma
#         self.G_model = copy.deepcopy(self.nn)
#         self.index = 0
#         self.temp_client = copy.deepcopy(self.nns)
#         self.temp_server = copy.deepcopy(self.nn)
#
#         for i in range(self.args.K):
#             temp = copy.deepcopy(self.nn)
#             temp.name = self.args.clients[i]
#             self.nns.append(temp)
#
#     def server(self):
#         for t in range(self.args.GE):
#             print('server round epoch:', t + 1, ':')
#             # sampling
#             m = np.max([int(self.args.C * self.args.K), 1])
#             index = random.sample(range(0, self.args.K), m)  # st
#             # dispatch
#             self.dispatch(index)
#             # local updating
#             self.client_update(index, t)
#             # aggregation
#             self.aggregation(index)
#
#     def aggregation(self, index):
#         s = 0
#         for j in index:
#             # normal
#             s += self.nns[j].len
#
#         params = {}
#         for k, v in self.nns[0].named_parameters():
#             params[k] = torch.zeros_like(v.data)
#
#         for j in index:
#             for k, v in self.nns[j].named_parameters():
#                 params[k] += v.data * (self.nns[j].len / s)
#
#         for k, v in self.nn.named_parameters():
#             v.data = params[k].data.clone()
#
#     def client_update(self, index, global_round):
#         for k in index:
#             # print('client update epoch:k:', k)
#             self.nns[k] = FedAP_train(self.args, self.nns[k], self.nn, self.train_set[k], global_round)
#             # 此处需要修改
#             self.nns[k].len = 10
#
#     def dispatch(self, index):
#         for j in index:
#             for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
#                 old_params.data = new_params.data.clone()
#
#     def global_test(self):
#         model = self.nn
#         model.eval()
#         model.name = self.args.clients[1]
#         test(self.args, self.test_set, self.Scale, model)
#         # for client in self.args.clients:
#         #     model.name = client
#         #     test(self.args, self.test_set, self.Scale, model)
#         # print("self.train_set", self.train_set)
#
#     # def local_test(self):
#     #     for i in range(len(self.nns)):
#     #         model = self.nns[i]
#     #         model.eval()
#     #         for client in self.args.clients:
#     #             model.name = client
#     #             print("test:", i)
#     #         test(self.args, self.train_set[i], self.Scale, model)
#
#     def e(self, x):
#         return math.exp(-x / self.sigma) / self.sigma
#
#     def train(self):
#         for i in range(self.args.GE + 1):
#             self.send_model()
#             for model in self.nns:
#                 model.train()  # ????
#             #   model = FedAP_train(model)是不是应该改+
#
#     def send_model(self, index):
#         for i in index:
#             c = copy.deepcopy(self.nns[i])
#             mu = copy.deepcopy(self.G_model)
#             for param in mu.parameters():
#                 param.data.zero_()
#             coef = torch.zeros(index)
#             for j in index:
#                 if i != j:
#                     mw = copy.deepcopy(self.nns[j])
#                     weights_i = torch.cat([p.data.view(-1) for p in c.model.parameters()], dim=0)
#                     weights_j = torch.cat([p.data.view(-1) for p in mw.parameters()], dim=0)
#                     sub = (weights_i - weights_j).view(-1)
#                     sub = torch.dot(sub, sub)
#                     coef[j] = self.alphaK * self.e(sub)
#                 else:
#                     coef[j] = 0
#             coef = 1 - torch.sum(coef)
#             # print(i, coef)
#
#             for j in index:
#                 mw = self.nns[j]
#                 for param, param_j in zip(mu.parameters(), mw.parameters()):
#                     param.data += coef[j] * param_j
#
#     def get_rand_index(self):
#         m = np.max([int(self.args.C * self.args.K), 1])
#         self.index = random.sample(range(0, self.args.K), m)