__author__ = 'yihanjiang'
'''
This is the utility function to compute FLOP, only can be used for TurboAE-CNN. RNN is not supported and hand-computed.
Need to install thop.
'''

import torch
import torch.nn.functional as F

from interleavers import Interleaver, DeInterleaver

# Compute the FLOP
from thop import profile

#######################################################
# DTA Encocder, with rate 1/3, CNN-1D same shape only
#######################################################
from cnn_utils import SameShapeConv1d
from encoders import ENCBase
class ENC_interCNN(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interCNN, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_3       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int  = self.interleaver(inputs)

        x_p2       = self.enc_cnn_3(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes


##################################################
# DTA Decoder with rate 1/3
# 1D CNN same shape decoder
##################################################

from encoders import SameShapeConv1d
class DEC_LargeCNN(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeCNN, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )

            self.dec2_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )
            self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_cnns[idx] = torch.nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = torch.nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final

class DEC_LargeRNN(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeRNN, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        self.dec1_rnns      = torch.nn.ModuleList()
        self.dec2_rnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                                        num_layers=2, bias=True, batch_first=True,
                                                        dropout=args.dropout, bidirectional=True)
            )

            self.dec2_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=args.dropout, bidirectional=True)
            )

            self.dec1_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return inputs

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_rnns[idx] = torch.nn.DataParallel(self.dec1_rnns[idx])
            self.dec2_rnns[idx] = torch.nn.DataParallel(self.dec2_rnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            if self.args.is_parallel:
                self.dec1_rnns[idx].module.flatten_parameters()
            x_dec, _   = self.dec1_rnns[idx](x_this_dec)
            x_plr      = self.dec_act(self.dropout(self.dec1_outputs[idx](x_dec)))

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            if self.args.is_parallel:
                self.dec2_rnns[idx].module.flatten_parameters()
            x_dec, _   = self.dec2_rnns[idx](x_this_dec)
            x_plr      = self.dec_act(self.dropout(self.dec2_outputs[idx](x_dec)))

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        if self.args.is_parallel:
            self.dec1_rnns[self.args.num_iteration - 1].module.flatten_parameters()

        x_dec, _   = self.dec1_rnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec_act(self.dropout(self.dec1_outputs[self.args.num_iteration - 1](x_dec)))

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        if self.args.is_parallel:
            self.dec2_rnns[self.args.num_iteration - 1].module.flatten_parameters()

        x_dec, _   = self.dec2_rnns[self.args.num_iteration - 1](x_this_dec)

        x_plr      = self.dec_act(self.dropout(self.dec2_outputs[self.args.num_iteration - 1](x_dec)))

        logit      = self.deinterleaver(x_plr)

        final      = torch.sigmoid(logit)

        return final




import torch
import numpy as np
from get_args import get_args

from numpy import arange
from numpy.random import mtrand


identity = str(np.random.random())[2:8]
print('[ID]', identity)

args = get_args()
print(args)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# setup interleaver.
if args.is_interleave == 1:
    seed = np.random.randint(0, 1)
    rand_gen = mtrand.RandomState(seed)
    p_array = rand_gen.permutation(arange(args.block_len))

elif args.is_interleave == 0:
    p_array = range(args.block_len)
else:
    seed = np.random.randint(0, args.is_interleave)
    rand_gen = mtrand.RandomState(seed)
    p_array = rand_gen.permutation(arange(args.block_len))
    print('using random interleaver', p_array)


model = DEC_LargeRNN(args, p_array)
#model = DEC_LargeCNN(args, p_array)
#model = ENC_interCNN(args, p_array)

flops, params = profile(model, input_size=(1,100,3))
print(flops, params)
