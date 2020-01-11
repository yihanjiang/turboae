__author__ = 'yihanjiang'
import torch
import torch.nn.functional as F

from cnn_utils import SameShapeConv1d

##############################################
# STE implementation
##############################################

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):

        enc_value_limit = 1.0
        enc_quantize_level = 2.0

        ctx.save_for_backward(inputs)
        ctx.enc_value_limit = enc_value_limit
        ctx.enc_quantize_level = enc_quantize_level

        x_lim_abs   =  enc_value_limit
        x_lim_range =  2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if enc_quantize_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(enc_quantize_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_output[input>ctx.enc_value_limit]=0
        grad_output[input<-ctx.enc_value_limit]=0
        grad_input = grad_output.clone()

        return grad_input, None, None




class Modulation(torch.nn.Module):
    def __init__(self, args):
        super(Modulation, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.args = args

        self.mod_layer = SameShapeConv1d(num_layer=args.mod_num_layer, in_channels=args.mod_rate,
                                          out_channels= args.mod_num_unit, kernel_size = 1, no_act = False)
        self.mod_final = SameShapeConv1d(num_layer=1, in_channels=args.mod_num_unit,
                                          out_channels= 2, kernel_size = 1,  no_act = True)


    def forward(self, inputs):
        # Input has shape (B, L, R)
        # output has shape (B, L * mod_rate, 2), last dimension is real, imag.

        inputs_flatten = inputs.view(self.args.batch_size, int(self.args.block_len * self.args.code_rate_n / self.args.mod_rate), self.args.mod_rate)
        mod_symbols = self.mod_final(self.mod_layer(inputs_flatten))

        if self.args.mod_pc == 'qpsk':
            this_mean    = torch.mean(mod_symbols)
            this_std     = torch.std(mod_symbols)
            mod_symbols  = (mod_symbols - this_mean)/this_std
            stequantize  = STEQuantize.apply
            outputs = stequantize(mod_symbols)
        elif self.args.mod_pc == 'symbol_power':
            this_mean    = torch.mean(torch.mean(mod_symbols, dim=2), dim=0)
            new_symbol = mod_symbols.permute(0,2,1)
            new_symbol_shape = new_symbol.shape
            this_std = torch.std(new_symbol.view(new_symbol_shape[0]*new_symbol_shape[1],new_symbol_shape[2]), dim=0)

            this_mean = this_mean.unsqueeze(0).unsqueeze(2)
            this_std = this_std.unsqueeze(0).unsqueeze(2)
            outputs  = (mod_symbols - this_mean)/this_std

        elif self.args.mod_pc == 'block_power':
            this_mean    = torch.mean(mod_symbols)
            this_std     = torch.std(mod_symbols)
            outputs  = (mod_symbols - this_mean)/this_std

        return outputs


class DeModulation(torch.nn.Module):
    def __init__(self, args):
        super(DeModulation, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.args = args

        self.demod_layer = SameShapeConv1d(num_layer=args.demod_num_layer, in_channels=2,
                                          out_channels= self.args.demod_num_unit, kernel_size = 1)
        self.demod_final = SameShapeConv1d(num_layer=1, in_channels=args.demod_num_unit,
                                          out_channels= args.mod_rate, kernel_size = 1,  no_act = True)

    def forward(self, inputs):
        # Input has shape (B, L * mod_rate, 2)
        # output has shape (B, L, R) , last dimension is real, imag.
        demod_symbols = self.demod_final(self.demod_layer(inputs))
        demod_codes = demod_symbols.reshape(self.args.batch_size, self.args.block_len, self.args.code_rate_n)

        return demod_codes