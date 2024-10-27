import copy
from PREPROCESS.preprocess import *
from MY_MODEL.mlp import *
from OBJECT.EWC import EWC
from RESULT.result_figure import scatter_plot
from RESULT.calculate_result import count_error
from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import chain


def ann_train(args, feature, label):
    # 封装ANN训练的文件，返回loss的列表和模型本身
    net = ANN(args, 'ANN')
    net.train()
    feature = feature.squeeze(-1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    loss = 0
    loss_all = []
    print("ANN-training")
    for i in range(500):
        pred = net(feature)
        loss = loss_func(pred.squeeze(-1), label)
        optimizer.zero_grad()
        loss.backward()
        loss_all.append(loss.detach().numpy())
        optimizer.step()
    return net, loss_all


def test_ann(test, net, loss):
    # 封装的测试文件
    net.eval()
    # print('loss_all:', loss)
    plt.plot(np.linspace(start=1, stop=len(loss), num=len(loss)), loss)
    feature, label = spilt_feature_Label(test)
    pre = net(feature)
    pre = pre.detach().numpy()
    plt.show()
    Scale = data_process(get_data())[1]
    Test = Scale.inverse_transform(test)
    label = Test[:, 8]
    new_data = np.concatenate((feature, pre), axis=1)
    new_data = Scale.inverse_transform(new_data)
    # print('new_data:', new_data)
    pre = new_data[:, 8]
    # print('pre:', pre)
    # print('label', label)
    scatter_plot(pre, label)
    count_error(pre, label)


def FedA_train(args, L_model, G_model, data, global_round):
    feature, Label = spilt_feature_Label(data)
    L_model.len = len(feature)
    L_model = copy.deepcopy(G_model)
    L_model.train()
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(L_model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(L_model.parameters(), lr=lr,
                                    weight_decay=args.weight_decay)
    # print('FedAvg training')
    feature = feature.squeeze(-1)
    Loss_func = nn.MSELoss()
    loss = 0
    for epoch in range(args.LE):
        pred = L_model(feature)
        optimizer.zero_grad()
        loss = Loss_func(pred.squeeze(-1), Label)
        # print(loss)
        loss.backward()
        optimizer.step()
    return L_model


def FedP_train(args, L_model, G_model, data, global_round):
    feature, label = spilt_feature_Label(data)
    L_model.len = len(feature)
    L_model = copy.deepcopy(G_model)
    L_model.train()
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(L_model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(L_model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    # print('FedProx training...')
    feature.squeeze(-1)
    loss_function = nn.MSELoss()
    loss = 0
    for epoch in range(args.LE):
        pred = L_model(feature)
        optimizer.zero_grad()
        pred = pred.squeeze(-1)
        # compute proximal_term
        proximal_term = 0.0
        for w, w_t in zip(L_model.parameters(), G_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        loss = loss_function(pred, label) + (args.mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
        # print('epoch', epoch, ':', loss.item())
    return L_model


def ewc_train(model: nn.Module, optimizer: torch.optim, data,
              ewc: EWC, importance: float):
    model.train()
    feature, label = spilt_feature_Label(data)
    loss_all = []
    loss_func = nn.MSELoss()
    # print("EWC training")
    # for _ in tqdm(range(50)):
    for _ in range(50):
        optimizer.zero_grad()
        pre = model(feature)
        pre = pre.squeeze(-1)
        loss = loss_func(pre, label) + importance * ewc.penalty(model)
        # print(importance * ewc.penalty(model))
        # 这里是不是少了一个data？
        loss.backward()
        loss_all.append(loss)
        optimizer.step()
    # print('loss:', loss_all)
    return model


def normal_train(model: nn.Module, optimizer: torch.optim, data):
    data = convert_to_tensor(data)
    loss_func = nn.MSELoss()
    model.train()
    epoch_loss = []
    feature, label = spilt_feature_Label(data)
    feature.squeeze(-1)
    # print('normal_train')
    for _ in range(50):
        optimizer.zero_grad()
        pre = model(feature)
        pre = pre.squeeze(-1)
        loss = loss_func(pre, label)
        epoch_loss.append(loss)
        loss.backward()
        optimizer.step()
    # print('loss:', epoch_loss)
    return model


def FedE_train(args, L_model, G_model, importance, data, label):
    # 初始化模型和数据
    L_model.len = len(data)
    L_model = copy.deepcopy(G_model)
    lr = args.lr
    optimizer = torch.optim.Adam(L_model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    # print('FedEWC training')
    if label == True:
        model = normal_train(L_model, optimizer, data)
    else:
        model = ewc_train(L_model, optimizer, data, EWC(L_model, data), importance)
    return model


def FedAP_train(args, L_model, G_model, data):
    feature, label = spilt_feature_Label(data)
    feature.squeeze(-1)
    L_model.len = len(feature)
    L_model = copy.deepcopy(G_model)
    L_model.train()

    lr = args.lr
    loss = 0
    optimizer = torch.optim.Adam(L_model.parameters(), lr=lr,
                                 weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()

    for epoch in range(args.LE):
        pred = L_model(feature)
        pred = pred.squeeze(-1)
        loss = loss_function(pred, label)

        gm = torch.cat([p.data.view(-1) for p in L_model.parameters()], dim=0)
        pm = torch.cat([p.data.view(-1) for p in G_model.parameters()], dim=0)
        loss += 0.5*args.lamda / args.alphaK * torch.norm(gm-pm, p=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return L_model


def test(args, test_set, Scale, net):
    net.eval()
    Feature, Label = spilt_feature_Label(test_set)
    with torch.no_grad():
        pred = net(Feature)
    pred = np.array(pred.detach().numpy())
    # print('pred:', pred)
    # print('label:', Label)
    new_data = np.concatenate((Feature, pred), axis=1)
    new_data = Scale.inverse_transform(new_data)
    pred = new_data[:, 8]
    data = Scale.inverse_transform(test_set)
    Label = data[:, 8]
    # print('pred:', pred)
    # print('label:', Label)
    count_error(pred, Label)
    scatter_plot(pred, Label)

