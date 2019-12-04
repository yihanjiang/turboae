__author__ = 'yihanjiang'

import torch

# STE implementation
class STEQuantize(torch.autograd.Function):
    #self.args.fb_quantize_limit, self.args.fb_quantize_level
    @staticmethod
    def forward(ctx, inputs, quant_limit, quant_level):

        ctx.save_for_backward(inputs)

        x_lim_abs  = quant_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if quant_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((quant_level - 1.0)/x_lim_range)) * x_lim_range/(quant_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # let's see what happens....
        # grad_output[torch.abs(input)>1.5]=0
        # grad_output[torch.abs(input)<0.5]=0

        grad_output[input>1.0]=0
        grad_output[input<-1.0]=0

        grad_output = torch.clamp(grad_output, -0.25, +0.25)

        grad_input = grad_output.clone()

        return grad_input, None, None, None