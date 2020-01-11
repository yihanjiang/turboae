from interleavers import Interleaver, DeInterleaver

import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np

##################################################
# DTA Decoder with rate 1/3
# RNN version
##################################################

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

##################################################
# DTA Decoder with rate 1/3
# 1D CNN same shape decoder
##################################################

from encoders import SameShapeConv1d, DenseSameShapeConv1d
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

        if self.args.encoder == 'TurboAE_rate3_cnn':
            CNNLayer = SameShapeConv1d
        else:
            CNNLayer = DenseSameShapeConv1d


        for idx in range(args.num_iteration):
            self.dec1_cnns.append(CNNLayer(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )

            self.dec2_cnns.append(CNNLayer(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
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

        if self.args.is_variable_block_len:
            block_len = received.shape[1]
            # reset interleaver
            if self.args.is_interleave != 0:           # fixed interleaver.
                seed = np.random.randint(0, self.args.is_interleave)
                rand_gen = mtrand.RandomState(seed)
                p_array = rand_gen.permutation(arange(block_len))
                self.set_interleaver(p_array)
        else:
            block_len = self.args.block_len

        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, block_len, self.args.num_iter_ft)).to(self.this_device)

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



##################################################
# DTA Decoder with rate 1/3
# 1D CNN same shape decoder
##################################################

from encoders import SameShapeConv1d
class DEC_LargeCNN2Int(torch.nn.Module):
    def __init__(self, args, p_array1, p_array2):
        super(DEC_LargeCNN2Int, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver1          = Interleaver(args, p_array1)
        self.deinterleaver1        = DeInterleaver(args, p_array1)

        self.interleaver2          = Interleaver(args, p_array2)
        self.deinterleaver2        = DeInterleaver(args, p_array2)

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
        pass

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int1 = self.interleaver1(r_sys)
        r_sys_int2 = self.interleaver2(r_sys)

        r_par1    = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            prior      = self.interleaver1(prior)
            x_this_dec = torch.cat([r_sys_int1, r_par1, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.deinterleaver1(x_plr)
            x_plr_int  = self.interleaver2(x_plr_int)

            x_this_dec = torch.cat([r_sys_int2, r_par2, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver2(x_plr)

        # last round
        prior      = self.interleaver1(prior)
        x_this_dec = torch.cat([r_sys_int1, r_par1, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.deinterleaver1(x_plr)
        x_plr_int  = self.interleaver2(x_plr_int)

        x_this_dec = torch.cat([r_sys_int2, r_par2, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver2(x_plr))

        return final




# experimental cnn 2d
from encoders import SameShapeConv2d, DenseSameShapeConv2d
from interleavers import DeInterleaver2D, Interleaver2D

class DEC_LargeCNN2D(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeCNN2D, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        if self.args.encoder == 'TurboAE_rate3_cnn2d_dense':
            CNN2d = DenseSameShapeConv2d
        else:
            CNN2d = SameShapeConv2d

        self.interleaver          = Interleaver2D(args, p_array)
        self.deinterleaver        = DeInterleaver2D(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_cnns.append(CNN2d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )

            self.dec2_cnns.append(CNN2d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            )
            self.dec1_outputs.append(CNN2d(1, args.dec_num_unit, args.num_iter_ft , kernel_size=1))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(CNN2d(1, args.dec_num_unit,1 , kernel_size=1, no_act = True))
            else:
                self.dec2_outputs.append(CNN2d(1, args.dec_num_unit,args.num_iter_ft , kernel_size=1))


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
        received = received.view(self.args.batch_size, self.args.img_size, self.args.img_size, self.args.code_rate_n)
        received = received.permute(0, 3, 1, 2)
        # Turbo Decoder
        r_sys     = received[:,0,:, :].view((self.args.batch_size, 1, self.args.img_size, self.args.img_size))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,1,:, :].view((self.args.batch_size, 1, self.args.img_size, self.args.img_size))
        r_par2    = received[:,2,:, :].view((self.args.batch_size, 1, self.args.img_size, self.args.img_size))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.num_iter_ft, self.args.img_size, self.args.img_size)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 1)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 1)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 1)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 1)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))
        final      = final.view(self.args.batch_size, self.args.code_rate_k, self.args.block_len)
        final      = final.permute(0,2,1)

        return final


# experimental cnn 2d without interleaver

class DEC_CNN2D(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_CNN2D, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        if self.args.encoder == 'TurboAE_rate3_cnn2d_dense':
            CNN2d = DenseSameShapeConv2d
        else:
            CNN2d = SameShapeConv2d

        self.dec = CNN2d(num_layer=args.dec_num_layer, in_channels=3,
                         out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
        self.output = CNN2d(1, args.dec_num_unit, 1, kernel_size=1)


    def set_parallel(self):
        pass

    def set_interleaver(self, p_array):
        pass

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        received = received.view(self.args.batch_size, self.args.img_size, self.args.img_size, self.args.code_rate_n)
        received = received.permute(0, 3, 1, 2)

        x = self.dec(received)
        x = self.output(x)

        final      = torch.sigmoid(x)
        final      = final.view(self.args.batch_size, self.args.code_rate_k, self.args.block_len)
        final      = final.permute(0,2,1)

        return final



##################################################
# DTA Decoder with rate 1/2
##################################################
class DEC_LargeRNN_rate2(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeRNN_rate2, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec1_rnns      = torch.nn.ModuleList()
        self.dec2_rnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):

            self.dec1_rnns.append(torch.nn.GRU(1 + args.num_iter_ft,  args.dec_num_unit,
                                               num_layers=2, bias=True, batch_first=True,
                                               dropout=args.dropout, bidirectional=True))

            self.dec2_rnns.append(torch.nn.GRU(1+ args.num_iter_ft,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=args.dropout, bidirectional=True))

            self.dec1_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_rnns[idx] = torch.nn.DataParallel(self.dec1_rnns[idx])
            self.dec2_rnns[idx] = torch.nn.DataParallel(self.dec2_rnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def forward(self, received):

        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_int     = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))

        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):

            x_this_dec = torch.cat([r_sys,  prior], dim = 2)
            x_dec, _   = self.dec1_rnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_int, x_plr_int ], dim = 2)
            x_dec, _   = self.dec2_rnns[idx](x_this_dec)
            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,  prior], dim = 2)
        x_dec, _   = self.dec1_rnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_int, x_plr_int ], dim = 2)
        x_dec, _   = self.dec2_rnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final
##################################################
# Rate 1/2 CNN
##################################################

class DEC_LargeCNN_rate2(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeCNN_rate2, self).__init__()
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
        r_sys       = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int   = self.interleaver(r_sys)
        r_par       = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par_deint = self.deinterleaver(r_par)

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys,r_par_deint, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par_deint, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final

##################################################
# NeuralBCJR algorithm
##################################################
class CNN_decoder_rate3(torch.nn.Module):
    def __init__(self, args, p_array):
        super(CNN_decoder_rate3, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.dec_cnn = SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=3,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)


        self.final   = torch.nn.Linear(args.dec_num_unit, 1)

    def set_parallel(self):
        pass

    def set_interleaver(self, p_array):
        pass

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        x_plr      = self.dec_cnn(received)
        final      = torch.sigmoid(self.final(x_plr))

        return final



##################################################
# NeuralBCJR algorithm
##################################################
class NeuralTurbofyDec(torch.nn.Module):
    def __init__(self, args, p_array):
        super(NeuralTurbofyDec, self).__init__()

        self.args             = args

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec_rnn  = torch.nn.GRU(args.code_rate_n + args.num_iter_ft - 1 , args.dec_num_unit, num_layers=2, bias=True, batch_first=True,
                                   dropout=args.dropout, bidirectional=True)
        self.dec_out = torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft)

        self.dec_final = torch.nn.Linear(args.num_iter_ft, 1)

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def enc_act(self, inputs):
        if self.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.enc_act == 'elu':
            return F.elu(inputs)
        elif self.enc_act == 'relu':
            return F.relu(inputs)
        elif self.enc_act == 'selu':
            return F.selu(inputs)
        elif self.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        else:
            return inputs

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)

    def forward(self, inputs):
        inputs = inputs.type(torch.FloatTensor).to(self.device)
        ##############################################################
        #
        # Neural Turbo Decoder
        #
        ##############################################################
        input_shape = inputs.shape
        # Turbo Decoder
        r_sys     = inputs[:,:,0].view((input_shape[0], self.args.block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = inputs[:,:,1].view((input_shape[0], self.args.block_len, 1))
        r_par2    = inputs[:,:,2].view((input_shape[0], self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((input_shape[0], self.args.block_len, self.args.num_iter_ft)).to(self.device)

        for idx in range(self.args.num_iteration - 1):

            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)
            x_dec, _   = self.dec_rnn(x_this_dec)
            x_plr      = self.dec_out(x_dec)

            if not self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int,r_par2, x_plr_int ], dim = 2)

            x_dec, _   = self.dec_rnn(x_this_dec)
            x_plr      = self.dec_out(x_dec)

            if not self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)
        x_dec, _   = self.dec_rnn(x_this_dec)
        x_plr      = self.dec_out(x_dec)

        if not self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int,r_par2, x_plr_int ], dim = 2)
        x_dec, _   = self.dec_rnn(x_this_dec)
        x_dec      = self.dec_out(x_dec)
        x_final    = self.dec_final(x_dec)
        x_plr      = torch.sigmoid(x_final)

        final      = self.deinterleaver(x_plr)

        return final