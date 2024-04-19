import torch
from torch import tensor, randn, kron, eye, diag
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

from utils import unvectorize, vectorize


def kron2(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

activation_f = tanh
activation_f_diff = lambda in_tensor: 1 - tanh(in_tensor)**2
# d/dx (1- tanh(x)^2) = 2*tanh(x)*(1-tanh(x)**2)
activation_f_diff2 = lambda in_tensor: -2 * tanh(in_tensor) * (1 - tanh(in_tensor)**2)

#activation_f = lambda x: x
#activation_f_diff = lambda x: x
#activation_f_diff2 = lambda x: x

mat1 = tensor([[0.5, 0.7],
               [0.25, 0.2],
               [0.1, 0.3]], requires_grad=True)

mat2 = tensor([[1.],
               [1.]])

mat3 = tensor([[1., 1.]])

x = tensor([[2., 3., 5.],
            [1., 2., 3.]])


def layer1(x, mat1_):
    return x @ unvectorize(mat1_, (3,2))


def model(mat1):
    return mat3 @ (activation_f(layer1(x, mat1)) @ mat2)


inputs = (vectorize(mat1),)

loss = mat3 @ (activation_f(x @ mat1) @ mat2)
loss.backward(retain_graph=True)

print("Gradient of mat1")
print(mat1.grad)
print(x.t() @ ((mat3.t() @ mat2.t()) * activation_f_diff(x @ mat1)))
# vectorized gradient of mat1.
#print(kron2(eye(1), x.t()) @ vectorize(mat2.t() * activation_f_diff(x @ mat1)))


mat1.grad.zero_()
hessian_result = hessian(model, inputs)
print("Torch hessian:")
print(hessian_result[0][0])
out_layer1 = layer1(x, inputs[0])
out = activation_f(out_layer1)
#print(kron2(eye(2), x.t()))

# gives correct result:
prev_layers = vectorize(mat3.t() @ mat2.t() * activation_f_diff2(x @ mat1))
print(kron2(eye(2), x.t()) @ diag(prev_layers) @ kron2(eye(2), x))

