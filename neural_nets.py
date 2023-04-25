import torch.nn as nn

'''
Authored by Gary Lvov
'''

class ShallowerSparsePointAssocNetwork(nn.Module):
    def __init__(self, input_size=171, output_size=96):
        super(ShallowerSparsePointAssocNetwork, self).__init__()
        intersize_1 = 159
        intersize_2 = 128

        self.input_layer = nn.Linear(input_size, intersize_1)
        self.input_activation = nn.ReLU()

        self.hidden_layer_1 = nn.Linear(intersize_1, intersize_2)
        self.hidden_activation_1 = nn.ReLU()

        self.output_layer = nn.Linear(intersize_2, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)

        x = self.hidden_layer_1(x)
        x = self.hidden_activation_1(x)

        x = self.output_layer(x)
        return x

class DeeperSparsePointAssocNetwork(nn.Module):
    def __init__(self, input_size=171, output_size=96):
        super(DeeperSparsePointAssocNetwork, self).__init__()
        intersize_1 = 159
        intersize_2 = 128
        intersize_3 = 100
        intersize_4 = 80
        intersize_5 = 60
        intersize_6 = 45
        intersize_7 = 70

        self.input_layer = nn.Linear(input_size, intersize_1)
        self.input_activation = nn.ReLU()

        self.hidden_layer_1 = nn.Linear(intersize_1, intersize_2)
        self.hidden_activation_1 = nn.ReLU()

        self.hidden_layer_2 = nn.Linear(intersize_2, intersize_3)
        self.hidden_activation_2 = nn.ReLU()

        self.hidden_layer_3 = nn.Linear(intersize_3, intersize_4)
        self.hidden_activation_3 = nn.ReLU()

        self.hidden_layer_4 = nn.Linear(intersize_4, intersize_5)
        self.hidden_activation_4 = nn.ReLU()

        self.hidden_layer_5 = nn.Linear(intersize_5, intersize_6)
        self.hidden_activation_5 = nn.ReLU()

        self.hidden_layer_6 = nn.Linear(intersize_6, intersize_7)
        self.hidden_activation_6 = nn.ReLU()

        self.output_layer = nn.Linear(intersize_7, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)

        x = self.hidden_layer_1(x)
        x = self.hidden_activation_1(x)

        x = self.hidden_layer_2(x)
        x = self.hidden_activation_2(x)

        x = self.hidden_layer_3(x)
        x = self.hidden_activation_3(x)

        x = self.hidden_layer_4(x)
        x = self.hidden_activation_4(x)

        x = self.hidden_layer_5(x)
        x = self.hidden_activation_5(x)

        x = self.hidden_layer_6(x)
        x = self.hidden_activation_6(x)

        x = self.output_layer(x)
        return x