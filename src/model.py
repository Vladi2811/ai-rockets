import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(6, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
