# If there are no activation functions between the linear layers, the
# diagonal blocks of the Hessian are zeros.
import torch
from torch import tensor, randn, eye, kron
from torch.autograd.functional import hessian

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

mat4 = tensor([[1., 1.]])

x = tensor([[1., 5., 3.],
            [2., 3., 4.]])


def layer1(x, mat1_):
    return x @ unvectorize(mat1_ ** 1, (3, 2))


def layer2(x, mat2_):
    return x @ unvectorize(mat2_, (2, 2))


def model(mat1_, mat2_):
    return mat4 @ (layer2(layer1(x, mat1_), mat2_) @ mat3)


inputs = (vectorize(mat1), vectorize(mat2))

out_layer2 = mat4 @ ((x @ mat1) @ mat2 @ mat3)
print(out_layer2)
loss = out_layer2
loss.backward()

out_layer1 = x @ mat1
print(mat2.grad)
# X^T @ (dL/dY) = X^T @ (mat4.T @ mat3.T)
print(out_layer1.t() @ mat4.t() @ mat3.t())

print(mat1.grad)
# # X^T @ (dL/dY) = X^T @ (mat4.T @ mat3.T @ mat2.T)
print(x.t() @ mat4.t() @ mat3.t() @ mat2.t())

hessian_l = hessian(model, inputs, strict=False)

# Hessian as matrix
#print(torch.cat(tuple(torch.cat(row_block, dim=1) for row_block in hessian_l)))

#print(hessian_l)
print("True hessian:")
print(hessian_l[0][1])

print(kron(eye(2), x.t()  @ mat4.t() @ mat3.t()) @ S_matrix(2))
