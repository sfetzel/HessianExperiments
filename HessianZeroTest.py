# If there are no activation functions between the linear layers, the
# diagonal blocks of the Hessian are zeros.
import torch
from torch import tensor, randn
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

mat1 = tensor([[1., 2.],
               [3., 4.],
               [5., 6.]], requires_grad=True)
mat1 = randn((3, 2), requires_grad=True)

mat2 = tensor([[7.],
               [8.]], requires_grad=True)
mat2 = randn((2, 1), requires_grad=True)

x = tensor([1., 5., 3.])

def layer1(x, mat1_):
    return x @ (mat1_ ** 1).reshape(3, 2)

def layer2(x, mat2_):
    return x @ (mat2_ ** 1).reshape(2, 1)


def model(mat1_, mat2_):
    return (layer2(layer1(x, mat1_), mat2_)).sum()


inputs = (mat1.reshape(-1), mat2.reshape(-1))
hessian_l = hessian(model, inputs, strict=False)

# Hessian as matrix
#print(torch.cat(tuple(torch.cat(row_block, dim=1) for row_block in hessian_l)))

# The off-diagonal blocks are non-zero.
# for a two layer network without activations, we have I * X as off-diagonals.

print(hessian_l)
