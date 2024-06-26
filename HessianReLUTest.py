import torch
from torch import tensor, randn
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid, relu

mat1 = tensor([[1., 2.],
               [3., 4.],
               [5., 6.]], requires_grad=True)
# If we use the provided values, the hessian will be zero, because
# tanh(..) will have all the same values.
mat1 = randn((3, 2), requires_grad=True)

mat2 = tensor([[7.],
               [8.]], requires_grad=True)
mat2 = randn((2, 1), requires_grad=True)

x = tensor([1., 5., 3.])

def layer1(x, mat1):
    return x @ mat1.reshape(3, 2)

def layer2(x, mat2):
    return x @ mat2.reshape(2, 1)

def model(mat1, mat2):
    return (relu(layer2(relu(layer1(x, mat1)), mat2))).sum()


inputs = (mat1.reshape(-1), mat2.reshape(-1))
out = model(*inputs)
out.backward()
print(hessian(model, inputs))
print(mat1.is_leaf)
print(mat1.grad)
print(mat2.grad)

