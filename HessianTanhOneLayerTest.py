import torch
from torch import tensor, randn
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

activation_f = tanh
activation_f_diff = lambda in_tensor: 1 - tanh(in_tensor)**2
# d/dx (1- tanh(x)^2) = 2*tanh(x)*(1-tanh(x)**2)
activation_f_diff2 = lambda in_tensor: -2 * tanh(in_tensor) * (1 - tanh(in_tensor)**2)

mat1 = tensor([[0.5],
               [0.25]], requires_grad=True)

x = tensor([[4., 5.]])


def layer1(x, mat1):
    return x @ mat1.reshape(2, 1)


def model(mat1):
    return (activation_f(layer1(x, mat1))).sum()


inputs = (mat1.reshape(-1),)

loss = (activation_f(x @ mat1)).sum()
loss.backward(retain_graph=True)

print(mat1.grad)
print((activation_f_diff(x @ mat1) @ x).t())


mat1.grad.zero_()
hessian_result = hessian(model, inputs)
print(hessian_result[0][0])
out_layer1 = layer1(x, inputs[0])
out = activation_f(out_layer1)
print(activation_f_diff2(out_layer1) * x.t() * x)
