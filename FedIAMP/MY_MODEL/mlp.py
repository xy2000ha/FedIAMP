from torch import nn
"""create mlp model"""


class ANN(nn.Module):
    def __init__(self, args, name):
        super(ANN, self).__init__()
        self.name = name
        self.len = 0
        self.masker = None
        self.fc1 = nn.Linear(args.input_dim, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        return x


class ANN2(nn.Module):
    def __init__(self):
        super(ANN2, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        return x
