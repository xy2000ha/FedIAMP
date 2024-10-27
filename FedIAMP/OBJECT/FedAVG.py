import copy
from MY_MODEL.mlp import ANN
from PREPROCESS.preprocess import *
from TRIAIN_FUNC.client_train import FedA_train, test


class FedAvg:
    def __init__(self, args):
        self.args = args
        self.nn = ANN(args=self.args, name='server')
        self.nns = []
        self.train_set, self.test_set = create_set1()
        self.Scale = data_process(get_data())[1]
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.GE):
            print('server round epoch:', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index, t)
            # aggregation
            self.aggregation(index)

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

    def client_update(self, index, global_round):  # update nn
        for k in index:
            # print('client update epoch:k:', k)
            self.nns[k] = FedA_train(self.args, self.nns[k], self.nn, self.train_set[k], global_round)
            # 此处需要修改
            self.nns[k].len = len(self.train_set[k])

    def global_test(self):
        model = self.nn
        model.eval()
        model.name = self.args.clients[1]
        test(self.args, self.test_set, self.Scale, model)
        # for client in self.args.clients:
        #     model.name = client
        #     test(self.args, self.test_set, self.Scale, model)
        # print("self.train_set", self.train_set)

    def local_test(self):
        # 功能是测试全局模型在每一个本地数据集的性能
        for i in range(len(self.nns)):
            model = self.nn
            model.eval()
            for client in self.args.clients:
                model.name = client
                print("test:", i)
            test(self.args, self.train_set[i], self.Scale, model)
