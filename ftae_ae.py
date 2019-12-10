__author__ = 'yihanjiang'
'''
11/28/19: from now on, only block delayed scheme are considered!

'''
import torch
import torch.nn.functional as F

from interleavers import Interleaver, DeInterleaver

from cnn_utils import SameShapeConv1d, DenseSameShapeConv1d

from ste import STEQuantize

# Common utilities of Feedback encoders.
class FB_encoder_base(torch.nn.Module):
    def power_constraint(self, x_input, quantize_limit = 1.0, quantize_level = 2):
        # power constraint can be continuous and discrete.
        # Now the implementation is continuous, with normalizing via single phase.
        # x_input has the shape (B, L, 1) (has to be!)
        x_input_shape = x_input.shape

        this_mean = torch.mean(x_input)
        this_std  = torch.std(x_input)
        x_input = (x_input-this_mean)*1.0 / this_std
        x_input_norm = x_input.view(x_input_shape)

        #'group_norm','group_norm_quantize'

        if self.args.channel_mode == 'block_norm':
            res = x_input_norm
        else:
            encoder_quantize = STEQuantize.apply
            res = encoder_quantize(x_input_norm, quantize_limit, quantize_level)

        return res


# Feedback Turbo AE Encoder
class CNN_encoder(FB_encoder_base):
    def __init__(self, args, input_size = 1, is_systematic_bit = False, is_interleave = False, p_array = []):
        super(CNN_encoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.args = args

        self.is_systematic_bit = is_systematic_bit
        self.is_interleave = is_interleave

        if self.is_interleave:
            self.interleaver = Interleaver(args, p_array)

        self.p_array = p_array

        # Encoder
        self.enc_cnn    = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=input_size,
                                   out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)
        self.enc_linear = torch.nn.Linear(args.enc_num_unit, 1)

    def forward(self, inputs):
        bpsk_x =  2.0*inputs - 1.0

        if self.is_systematic_bit:
            return bpsk_x

        elif not self.is_interleave:
            x     = self.enc_cnn(bpsk_x)
            x     = F.elu(self.enc_linear(x))
            code  = self.power_constraint(x)
            return code

        else:
            x_int = self.interleaver(inputs)
            x     = self.enc_cnn(x_int)
            x     = F.elu(self.enc_linear(x))
            code  = self.power_constraint(x)
            return code



# Feedback Turbo AE Decoder
class FTAE_decoder(torch.nn.Module):
    def __init__(self, args, p_array):
        super(FTAE_decoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        # interleaver
        self.p_array = p_array
        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        # Decoder
        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        if args.cnn_type =='dense':
            CNNModel = DenseSameShapeConv1d
        else:
            CNNModel = SameShapeConv1d

        for idx in range(args.num_iteration):
            if self.args.dec_type == 'turboae_cnn':
                self.dec1_cnns.append(CNNModel(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                      out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
                )

                self.dec2_cnns.append(CNNModel(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                      out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
                )
                self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

                if idx == args.num_iteration -1:
                    self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.code_rate_k))
                else:
                    self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))
            else: # RNN based

                self.dec1_cnns.append(torch.nn.GRU(2 + args.num_iter_ft, args.dec_num_unit,
                                      num_layers=args.dec_num_layer, bias=True, batch_first=True,
                                      dropout=0, bidirectional=True)
                )

                self.dec2_cnns.append(torch.nn.GRU(2 + args.num_iter_ft, args.dec_num_unit,
                                      num_layers=args.dec_num_layer, bias=True, batch_first=True,
                                      dropout=0, bidirectional=True)
                )
                self.dec1_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

                if idx == args.num_iteration -1:
                    self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.code_rate_k))
                else:
                    self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

    def forward(self, received_codes):
        received = received_codes.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            if self.args.dec_type == 'turboae_cnn':
                x_dec    = self.dec1_cnns[idx](x_this_dec)
            else:
                x_dec, _ = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            if self.args.dec_type == 'turboae_cnn':
                x_dec   = self.dec2_cnns[idx](x_this_dec)
            else:
                x_dec,_ = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        if self.args.dec_type == 'turboae_cnn':
            x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        else:
            x_dec, _  = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)

        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        if self.args.dec_type == 'turboae_cnn':
            x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        else:
            x_dec, _  = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final




class FTAE_Shareddecoder(torch.nn.Module):
    def __init__(self, args, p_array):
        super(FTAE_Shareddecoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        # interleaver
        self.p_array = p_array
        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        if args.cnn_type =='dense':
            CNNModel = DenseSameShapeConv1d
        else:
            CNNModel = SameShapeConv1d

        self.dec1_cnns     = CNNModel(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
        self.dec1_outputs  = torch.nn.Linear(args.dec_num_unit, args.num_iter_ft)
        self.dec2_cnns     = CNNModel(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
        self.dec2_outputs  = torch.nn.Linear(args.dec_num_unit, args.num_iter_ft)

        self.final_outputs = torch.nn.Linear(args.num_iter_ft, 1)

    def forward(self, received_codes):
        received = received_codes.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            x_dec    = self.dec1_cnns(x_this_dec)
            x_plr      = self.dec1_outputs(x_dec)

            x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec   = self.dec2_cnns(x_this_dec)

            x_plr      = self.dec2_outputs(x_dec)

            x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        final      = torch.sigmoid(self.final_outputs(self.deinterleaver(x_plr)))

        return final




class CNN_decoder(torch.nn.Module):
    def __init__(self, args):
        super(CNN_decoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.dec_cnn    = SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=args.code_rate_n,
                                              out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
        self.dec_output = torch.nn.Linear(args.dec_num_unit, args.code_rate_k)

    def forward(self, received_codes):
        x_dec     = self.dec_cnn(received_codes)
        x_dec     = torch.sigmoid(self.dec_output(x_dec))
        return x_dec

#######################################################################################################################
#
# Channel Autoencoder, for now only support code rate 1/3, and 2/3
#
#######################################################################################################################

class Channel_Feedback_rate3(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Channel_Feedback_rate3, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        # interleaver
        self.p_array = p_array

        if args.is_interleave:
            is_interleave = True
        else:
            is_interleave = False

        self.fwd_enc1 = CNN_encoder(args, input_size=1, is_systematic_bit=False, is_interleave=False)
        self.fwd_enc2 = CNN_encoder(args, input_size=3, is_systematic_bit=False, is_interleave=False)
        self.fwd_enc3 = CNN_encoder(args, input_size=5, is_systematic_bit=False,
                                    is_interleave=is_interleave, p_array = p_array)

        # args, input_size = 1, is_systematic_bit = False, is_interleave = False, p_array = []

        self.fb_enc1 = CNN_encoder(args, input_size=1, is_systematic_bit=False, is_interleave=False)
        self.fb_enc2 = CNN_encoder(args, input_size=2, is_systematic_bit=False, is_interleave=False)

        if args.dec_type == 'cnn':
            self.dec = CNN_decoder(args)
        elif args.dec_type in ['turboae_cnn','turboae_rnn']:
            self.dec = FTAE_decoder(args, p_array)
        elif args.dec_type == 'turboae_sharedcnn':
            self.dec = FTAE_Shareddecoder(args, p_array)
        else:
            print('unknown decoder type.')

    def forward(self, inputs, fwd_z, fb_z):

        block_len = inputs.shape[1]

        # Decouple feedbacks
        fwd_z1  = fwd_z[:,:,0].view((self.args.batch_size, block_len, 1))
        fwd_z2  = fwd_z[:,:,1].view((self.args.batch_size, block_len, 1))
        fwd_z3  = fwd_z[:,:,2].view((self.args.batch_size, block_len, 1))

        fb_z1  = fb_z[:,:,0].view((self.args.batch_size, block_len, 1))
        fb_z2  = fb_z[:,:,1].view((self.args.batch_size, block_len, 1))
        fb_z3  = fb_z[:,:,2].view((self.args.batch_size, block_len, 1)) # 3nd phase not used....

        # Code Phase 1
        x_1 = self.fwd_enc1(inputs)
        y_1 = x_1 + fwd_z1
        f_1 = self.fb_enc1(y_1)
        r_1 = f_1 + fb_z1

        # Code Phase 2
        if self.args.ignore_feedback:
            r_1 = r_1 * 0.0
        if self.args.ignore_prev_code:
            x_1 = x_1 * 0.0

        input_2 = torch.cat([inputs, r_1, x_1], dim=2)
        x_2 = self.fwd_enc2(input_2)
        y_2 = x_2 + fwd_z2
        y_input = torch.cat([y_1, y_2], dim=2)
        f_2 = self.fb_enc2(y_input)
        r_2 = f_2 + fb_z2

        # Code Phase 3
        if self.args.ignore_feedback:
            r_2 = r_2 * 0.0
        if self.args.ignore_prev_code:
            x_2 = x_2 * 0.0

        input_3 = torch.cat([inputs, r_1, x_1, r_2, x_2], dim=2)
        x_3 = self.fwd_enc3(input_3)
        y_3 = x_3 + fwd_z3

        codes = torch.cat([x_1, x_2, x_3], dim=2)

        received_codes = torch.cat([y_1, y_2, y_3], dim=2)
        final = self.dec(received_codes)
        return final, codes
