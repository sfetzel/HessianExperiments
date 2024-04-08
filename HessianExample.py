# from https://devcodef1.com/news/1174213/pytorch-hessian-computation-differences
# does not work properly
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = TwoLayerNet()

x = torch.randn(3, 10)
y = torch.randn(3, 2)

# Compute gradients
out = net(x)
loss = nn.MSELoss()(out, y)
net.zero_grad()
loss.backward()

# Compute Hessian
hessian = torch.tensor([[net.fc2.weight.grad.grad.item(), 0],
[0, net.fc2.weight.grad.grad.item()]])
print(hessian)