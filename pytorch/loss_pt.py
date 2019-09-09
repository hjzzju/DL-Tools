import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable

from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
#if all manipulation needed can be achieved by pytorch tensor function
# Loss is just a kind of pytorch layer 
class BaseLoss(nn.Module):

    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, input, target):
        #return loss
        pass
#for example
# class StableBCELoss(torch.nn.modules.Module):
#     def __init__(self):
#          super(StableBCELoss, self).__init__()
#     def forward(self, input, target):
#          neg_abs = - input.abs()
#          loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
#          return loss.mean()

#example for use self-defined backward
# class ScipyConv2dFunction(Function):
#     @staticmethod
#     def forward(ctx, input, filter):
#         result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
#         ctx.save_for_backward(input, filter)
#         return input.new(result)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, filter = ctx.saved_tensors
#         grad_output = grad_output.data
#         grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
#         grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')

#         return Variable(grad_output.new(grad_input)), \
#             Variable(grad_output.new(grad_filter))


# class ScipyConv2d(Module):

#     def __init__(self, kh, kw):
#         super(ScipyConv2d, self).__init__()
#         self.filter = Parameter(torch.randn(kh, kw))

#     def forward(self, input):
#         return ScipyConv2dFunction.apply(input, self.filter)