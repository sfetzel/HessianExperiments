import torch
from torch import tensor, randn, kron, eye, diag, diag_embed
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

from utils import vectorize, unvectorize

activation_f = tanh
activation_f_diff = lambda in_tensor: 1 - tanh(in_tensor)**2
# d/dx (1- tanh(x)^2) = 2*tanh(x)*(1-tanh(x)**2)
activation_f_diff2 = lambda in_tensor: -2 * tanh(in_tensor) * (1 - tanh(in_tensor)**2)

def kron2(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

def pointwise(A, B):
    assert A.shape == B.shape
    return A * B

mat1 = tensor([[1., 2.],
               [3., 4.],
               [5., 6.]], requires_grad=True)
# If we use the provided values, the hessian will be zero, because
# tanh(..) will have all the same values.
mat1 = randn((3, 2), requires_grad=True)

mat2 = tensor([[7., 9.],
               [8., 10.]], requires_grad=True)
#mat2 = randn((2, 1), requires_grad=True)

mat3 = tensor([[1.], [1.]])

x = tensor([[1., 2., 1.]])

def layer1(x, mat1):
    return x @ unvectorize(mat1, (3, 2))

def layer2(x, mat2):
    return x @ unvectorize(mat2, (2, 2))

def model(mat1, mat2):
    return layer2(activation_f(layer1(x, mat1)), mat2) @ mat3


inputs = (vectorize(mat1), vectorize(mat2))
out = model(*inputs)
out.backward()

out_layer1 = layer1(x, inputs[0])
activations_layer1 = activation_f(out_layer1)
out_layer2 = layer2(activations_layer1, inputs[1])

# second layer gradient
print("\nSecond layer gradient")
print(mat2.grad)
print(activations_layer1.T @ mat3.T)

# first layer gradient
print("\nFirst layer gradient")
print(mat1.grad)
print(x.t() @ (pointwise(mat3.t() @ mat2.t(), activation_f_diff(out_layer1))))

hessian_l = hessian(model, inputs)
# second layer hessian
print("\nSecond layer hessian")
# zero because there is no activation function.
print(hessian_l[1][1])

# first layer hessian
# derivative of inputs[1].t() * activation_f_diff(out_layer1) * x.t()
print("\nFirst layer hessian")
print(hessian_l[0][0])

activation_diag = diag(vectorize(mat3.t() @ mat2.t()) * vectorize(activation_f_diff2(out_layer1)))

print(kron2(eye(2), x.t()) @ activation_diag @ kron(eye(2), x))


