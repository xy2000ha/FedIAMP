from PREPROCESS.preprocess import *
from TRIAIN_FUNC.client_train import *
from PREPROCESS import args
from sklearn.preprocessing import MinMaxScaler
from OBJECT import FedAVG, FedPROX, FedEWC, FedAMP, Fed_I_AMP


def ann_solo(args):
    data = get_data()
    data, scale = data_process(data)
    # train, test = train_test_split(data, train_size=0.10, random_state=42)
    train = data[60:69, :]
    test = data[70:, :]
    train_feature, train_label = spilt_feature_Label(train)
    net, loss = ann_train(args, train_feature, train_label)
    test_ann(test, net, loss)
    # my_save_model(net)

def ann_central(args):
    data = get_data()
    data, scale = data_process(data)
    # train, test = train_test_split(data, train_size=0.10, random_state=42)
    train = data[:69, :]
    test = data[70:, :]
    train_feature, train_label = spilt_feature_Label(train)
    net, loss = ann_train(args, train_feature, train_label)
    test_ann(test, net, loss)
    # my_save_model(net)


def FedA(args):
    FedAvg = FedAVG.FedAvg(args)
    FedAvg.server()
    FedAvg.global_test()
    # FedAvg.local_test()
    return 0


def FedP(args):
    FedProx = FedPROX.FedProx(args)
    FedProx.server()
    FedProx.global_test()
    # FedProx.local_test()
    return 0

def FedE(args):
    FedEwc = FedEWC.FedEwc(args)
    FedEwc.my_server()
    # FedEwc.server()
    FedEwc.global_test()
    # FedEwc.global_train()
    # FedEwc.local_test()
    # FedEwc.local_train()
    return 0

def FedAP(args):
    FedAmp = FedAMP.FedAMP(args)
    FedAmp.servers()
    # FedAmp.global_test()
    FedAmp.local_test()

def FedIAP(args):
    Fediamp = Fed_I_AMP.Fed_I_Amp(args)
    Fediamp.my_servers()
    # Fediamp.global_test()
    Fediamp.local_test()


args = args.args_parser()

# print('ann_solo-training')
# ann_solo(args)

# print('ann_central-training')
# ann_central(args)

# print("FedAvg training")
# FedA(args)

# print("FedProx training")
# FedP(args)

# print("FedEWC training")
# FedE(args)

# print("FedAMP training")
# FedAP(args)

print("Fed_I_AMP training")
FedIAP(args)
