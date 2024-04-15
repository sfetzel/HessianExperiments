import torch
from torch import tensor, randn, kron, eye
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.nn.functional import tanh, sigmoid

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

mat2 = tensor([[7.],
               [8.]], requires_grad=True)
mat2 = randn((2, 1), requires_grad=True)

x = tensor([[1., 1., 3.]])

def layer1(x, mat1):
    return x @ mat1.reshape(3, 2)

def layer2(x, mat2):
    return x @ mat2.reshape(2, 1)

def model(mat1, mat2):
    return (layer2(activation_f(layer1(x, mat1)), mat2)).sum()


inputs = (mat1.reshape(-1), mat2.reshape(-1))
out = model(*inputs)
out.backward()

out_layer1 = layer1(x, inputs[0])
activations_layer1 = activation_f(out_layer1)
out_layer2 = layer2(activations_layer1, inputs[1])

# second layer gradient
print("\nSecond layer gradient")
print(mat2.grad)
print(activations_layer1)

# first layer gradient
print("\nFirst layer gradient")
print(mat1.grad)
print(inputs[1].t() * activation_f_diff(out_layer1) * x.t())

hessian_l = hessian(model, inputs)
# second layer hessian
print("\nSecond layer hessian")
print(hessian_l[1][1])
print(activation_f_diff2(out_layer2) * activations_layer1 * activations_layer1.t())

# first layer hessian
# derivative of inputs[1].t() * activation_f_diff(out_layer1) * x.t()
print("\nFirst layer hessian")
print(hessian_l[0][0])
# compare values of h_m with diagonal elements on hessian
h_m_1 = inputs[1].t() * activation_f_diff2(out_layer1) * x.t() * x.t()
#h_m_2 = activation_f_diff(out_layer2) * inputs[1].t() * activation_f_diff2(out_layer1) * x.t() * x.t()
h_m_2 = 0

print(h_m_1)

