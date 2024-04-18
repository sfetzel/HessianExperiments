# If there are no activation functions between the linear layers, the
# diagonal blocks of the Hessian are zeros.
import torch
from torch import tensor, randn, eye, kron
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

from utils import S_matrix, unvectorize, vectorize


def kron2(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])


mat1 = tensor([[1., 2.],
               [3., 4.],
               [5., 6.]], requires_grad=True)

mat2 = tensor([[7., 2.],
               [8., 3.]], requires_grad=True)

mat3 = tensor([[1.],
               [1.]])

x = tensor([[1., 5., 3.]]) # [2., 3., 4.]


def layer1(x, mat1_):
    return x @ unvectorize(mat1_ ** 1, (3, 2))


def layer2(x, mat2_):
    return x @ unvectorize(mat2_, (2, 2))


def model(mat1_, mat2_):
    return layer2(layer1(x, mat1_), mat2_) @ mat3


inputs = (vectorize(mat1), vectorize(mat2))

out_layer2 = (x @ mat1) @ mat2 @ mat3
print(out_layer2)
loss = out_layer2
loss.backward()

out_layer1 = x @ mat1
print(mat2.grad)
print(tensor([[1., 1.]]) * out_layer1.t())

print(mat1.grad)
print(tensor([[1., 1.]]) @ mat2.t() * x.t())

hessian_l = hessian(model, inputs, strict=False)

# Hessian as matrix
#print(torch.cat(tuple(torch.cat(row_block, dim=1) for row_block in hessian_l)))

#print(hessian_l)
print("True hessian:")
print(hessian_l[0][1])

print(kron2(kron(eye(2), tensor([[1., 1.]])) @ S_matrix(2), x.t()))
