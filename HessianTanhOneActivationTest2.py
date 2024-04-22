"""
Verifies the Hessian of a two layer neural network
where the second layer has an activation function but the first doesn't.
"""
import torch
from torch import tensor, randn, kron, eye, diag, diag_embed
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

from utils import vectorize, unvectorize, pointwise

activation_f = tanh
activation_f_diff = lambda in_tensor: 1 - tanh(in_tensor)**2
# d/dx (1- tanh(x)^2) = 2*tanh(x)*(1-tanh(x)**2)
activation_f_diff2 = lambda in_tensor: -2 * tanh(in_tensor) * (1 - tanh(in_tensor)**2)

def kron2(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])


mat1 = tensor([[1., 2.],
               [3., 4.],
               [5., 6.]], requires_grad=True)
# If we use the provided values, the hessian will be zero, because
# tanh(..) will have all the same values.
mat1 = randn((3, 2), requires_grad=True)

mat2 = tensor([[7., 9.],
               [8., 10.]], requires_grad=True)
mat2 = randn((2, 2), requires_grad=True)

mat3 = tensor([[1.], [1.]])

x = tensor([[1., 2., 1.]])

def layer1(x, mat1):
    return x @ unvectorize(mat1, (3, 2))

def layer2(x, mat2):
    return x @ unvectorize(mat2, (2, 2))

def model(mat1, mat2):
    return activation_f(layer2(layer1(x, mat1), mat2)) @ mat3


inputs = (vectorize(mat1), vectorize(mat2))
out = model(*inputs)
out.backward()

out_layer1 = layer1(x, inputs[0])
out_layer2 = layer2(out_layer1, inputs[1])
activations_2 = activation_f(out_layer2)

# second layer gradient
print("\nSecond layer gradient")
print(mat2.grad)
print(out_layer1.T @ pointwise(mat3.T, activation_f_diff(out_layer2)))

# first layer gradient
print("\nFirst layer gradient")
print(mat1.grad)
print(x.t() @ (pointwise(mat3.t(), activation_f_diff(out_layer2)) @ mat2.t()))

hessian_l = hessian(model, inputs)
# second layer hessian
print("\nSecond layer hessian")
print(hessian_l[1][1])
print(kron2(eye(2), out_layer1.t()) @ diag(vectorize(pointwise(mat3.t(), activation_f_diff2(out_layer2)))) @ kron2(eye(2), out_layer1))


# first layer hessian
# derivative of inputs[1].t() * activation_f_diff(out_layer1) * x.t()
print("\nFirst layer hessian")
print(hessian_l[0][0])

h0_diag = diag(vectorize(pointwise(mat3.t(), activation_f_diff2(out_layer2))))
# using: d/dW1 (X*W1*W2) = (W2^T kron x)
print(kron2(eye(2), x.t()) @ kron2(mat2, eye(1)) @ h0_diag @ kron2(mat2.t(), x))


