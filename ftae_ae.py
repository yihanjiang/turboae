__author__ = 'yihanjiang'
'''
11/28/19: from now on, only block delayed scheme are considered!

'''
import torch
import torch.nn.functional as F

from interleavers import Interleaver, DeInterleaver

from cnn_utils import SameShapeConv1d, DenseSameShapeConv1d

from ste import ChannelSTEQuantize
from ste import EncoderSTEQuantize

# Common utilities of Feedback encoders.
class FB_encoder_base(torch.nn.Module):
    def power_constraint(self, x_input):
        # power constraint can be continuous and discrete.
        # Now the implementation is continuous, with normalizing via single phase.
        # x_input has the shape (B, L, 1) (has to be!)
        x_input_shape = x_input.shape

        this_mean = torch.mean(x_input)
        this_std  = torch.std(x_input)
        x_input = (x_input-this_mean)*1.0 / this_std
        x_input_norm = x_input.view(x_input_shape)

        #'group_norm','group_norm_quantize'

        if self.args.channel_mode == 'group_norm':
            res = x_input_norm
        else:
            encoder_quantize = EncoderSTEQuantize.apply
            res = encoder_quantize(x_input_norm, self.args.fwd_quantize_limit, self.args.fwd_quantize_level)

        return res


# Feedback Turbo AE Encoder
class FTAE_encoder(FB_encoder_base):
    def __init__(self, args, p_array):
        super(FTAE_encoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        # interleaver
        self.p_array = p_array
        self.interleaver          = Interleaver(args, p_array)

        if self.args.cnn_type == 'dense':
            CNNModel = DenseSameShapeConv1d
        else:
            CNNModel = SameShapeConv1d

        # Encoder
        self.enc_cnn_1       = CNNModel(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)
        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = CNNModel(num_layer=args.enc_num_layer, in_channels=args.code_rate_k + 2,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_3       = CNNModel(num_layer=args.enc_num_layer, in_channels=args.code_rate_k + 4,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)
        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)



    def forward(self, inputs, fwd_noise, feedback_noise):
        ## Feedback TurboAE encoder
        ####################################
        # Phase 1
        ####################################
        bpsk_x =  2.0*inputs - 1.0
        x_1 = self.enc_cnn_1(bpsk_x)
        x_1 = F.elu(self.enc_linear_1(x_1))

        code_1 = self.power_constraint(x_1)

        # channel of Phase 1.
        r_1_rec = code_1 + fwd_noise[:, :, 0].view(self.args.batch_size, self.args.block_len, 1)

        # quantize received signal
        if self.args.rec_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            r_1 = Channel_quantize(r_1_rec, self.args.rec_quantize_limit, self.args.rec_quantize_level)
        else:
            r_1 = r_1_rec

        fbt_1_noise = r_1 + feedback_noise[:, :, 0].view(self.args.batch_size, self.args.block_len, 1)

        # quantize feedback signal
        if self.args.fb_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            fb_1 = Channel_quantize(fbt_1_noise, self.args.fb_quantize_limit, self.args.fb_quantize_level)
        else:
            fb_1 = fbt_1_noise

        if self.args.ignore_feedback:
            fb_1 = 0.01* torch.randn(fb_1.shape).to(self.this_device)

        if self.args.ignore_feedback_zero:
            fb_1 = 0.0 * fb_1

        if self.args.ignore_prev_code:
            code_1 = 0.0 * code_1

        if self.args.feed_noise_only:
            code_1 = fb_1 - code_1

        ####################################
        # Phase 2
        ####################################
        x_2 = torch.cat([bpsk_x ,fb_1, code_1], dim=2)
        x_2 = self.enc_cnn_2(x_2)
        x_2 = F.elu(self.enc_linear_2(x_2))
        code_2 = self.power_constraint(x_2)

        # channel of Phase 2.
        r_2_rec = code_2 + fwd_noise[:, :, 1].view(self.args.batch_size, self.args.block_len, 1)

        # quantize received signal
        if self.args.rec_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            r_2 = Channel_quantize(r_2_rec, self.args.rec_quantize_limit, self.args.rec_quantize_level)
        else:
            r_2 = r_2_rec
        # quantize feedback signal
        fbt_2_noise = r_2 + feedback_noise[:, :, 1].view(self.args.batch_size, self.args.block_len, 1)

        if self.args.fb_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            fb_2 = Channel_quantize(fbt_2_noise, self.args.fb_quantize_limit, self.args.fb_quantize_level)
        else:
            fb_2 = fbt_2_noise

        if self.args.ignore_feedback:
            fb_2 = 0.01* torch.randn(fb_2.shape).to(self.this_device)

        if self.args.ignore_feedback_zero:
            fb_2 = 0.0 * fb_2

        if self.args.ignore_prev_code:
            code_2 = 0.0 * code_2

        if self.args.feed_noise_only:
            code_2 = fb_2 - code_2

        # Phase 3
        bpsk_x_interleaved = self.interleaver(bpsk_x)
        fb_1_interleaved = self.interleaver(fb_1)
        fb_2_interleaved = self.interleaver(fb_2)
        code_1_interleaved = self.interleaver(code_1)
        code_2_interleaved = self.interleaver(code_2)

        x_3 = torch.cat([bpsk_x_interleaved ,fb_1_interleaved, fb_2_interleaved, code_1_interleaved, code_2_interleaved], dim=2)
        x_3 = self.enc_cnn_3(x_3)
        x_3 = F.elu(self.enc_linear_3(x_3))
        code_3 = self.power_constraint(x_3)

        # channel of Phase 3.
        r_3_rec = code_3 + fwd_noise[:, :, 2].view(self.args.batch_size, self.args.block_len, 1)

        # quantize received signal
        if self.args.rec_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            r_3 = Channel_quantize(r_3_rec, self.args.rec_quantize_limit, self.args.rec_quantize_level)
        else:
            r_3 = r_3_rec

        # FB Decoder
        received_codes = torch.cat([r_1, r_2, r_3], dim=2)

        return received_codes

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
            if self.args.codec == 'turboae_blockdelay_cnn':
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

            if self.args.codec == 'turboae_blockdelay_cnn':
                x_dec    = self.dec1_cnns[idx](x_this_dec)
            else:
                x_dec, _ = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            if self.args.codec == 'turboae_blockdelay_cnn':
                x_dec   = self.dec2_cnns[idx](x_this_dec)
            else:
                x_dec,_ = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        if self.args.codec == 'turboae_blockdelay_cnn':
            x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        else:
            x_dec, _  = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)

        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        if self.args.codec == 'turboae_blockdelay_cnn':
            x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        else:
            x_dec, _  = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final

#
class DeepCode_encoder(FB_encoder_base):
    def __init__(self, args):
        super(DeepCode_encoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        if args.cnn_type == 'dense':
            CNNModel = DenseSameShapeConv1d
        else:
            CNNModel = SameShapeConv1d

        # Encoder only use CNN
        self.enc_cnn_1       = CNNModel(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)
        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = CNNModel(num_layer=args.enc_num_layer, in_channels=args.code_rate_k + 2,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_3       = CNNModel(num_layer=args.enc_num_layer, in_channels=args.code_rate_k + 4,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)
        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)

    def forward(self, inputs, fwd_noise, feedback_noise):
        ## Feedback TurboAE encoder
        ####################################
        # Phase 1
        ####################################
        bpsk_x =  2.0*inputs - 1.0
        if self.args.is_systematic_code:
            code_1 = bpsk_x
        else:
            x_1 = self.enc_cnn_1(bpsk_x)
            x_1 = F.elu(self.enc_linear_1(x_1))
            code_1 = self.power_constraint(x_1)

        # channel of Phase 1.
        r_1_rec = code_1 + fwd_noise[:, :, 0].view(self.args.batch_size, self.args.block_len, 1)

        # quantize received signal
        if self.args.rec_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            r_1 = Channel_quantize(r_1_rec, self.args.rec_quantize_limit, self.args.rec_quantize_level)
        else:
            r_1 = r_1_rec

        fbt_1_noise = r_1 + feedback_noise[:, :, 0].view(self.args.batch_size, self.args.block_len, 1)

        # quantize feedback signal
        if self.args.fb_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            fb_1 = Channel_quantize(fbt_1_noise, self.args.fb_quantize_limit, self.args.fb_quantize_level)
        else:
            fb_1 = fbt_1_noise

        if self.args.ignore_feedback:
            fb_1 = 0.01* torch.randn(fb_1.shape).to(self.this_device)

        if self.args.ignore_feedback_zero:
            fb_1 = 0.0 * fb_1

        ####################################
        # Phase 2
        ####################################
        x_2 = torch.cat([bpsk_x ,fb_1, code_1], dim=2)
        x_2 = self.enc_cnn_2(x_2)
        x_2 = F.elu(self.enc_linear_2(x_2))
        code_2 = self.power_constraint(x_2)

        # channel of Phase 2.
        r_2_rec = code_2 + fwd_noise[:, :, 1].view(self.args.batch_size, self.args.block_len, 1)

        # quantize received signal
        if self.args.rec_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            r_2 = Channel_quantize(r_2_rec, self.args.rec_quantize_limit, self.args.rec_quantize_level)
        else:
            r_2 = r_2_rec
        # quantize feedback signal
        fbt_2_noise = r_2 + feedback_noise[:, :, 1].view(self.args.batch_size, self.args.block_len, 1)

        if self.args.fb_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            fb_2 = Channel_quantize(fbt_2_noise, self.args.fb_quantize_limit, self.args.fb_quantize_level)
        else:
            fb_2 = fbt_2_noise

        if self.args.ignore_feedback:
            fb_2 = 0.01* torch.randn(fb_2.shape).to(self.this_device)

        if self.args.ignore_feedback_zero:
            fb_2 = 0.0 * fb_2


        x_3 = torch.cat([bpsk_x,fb_1, fb_2, code_1, code_2], dim=2)
        x_3 = self.enc_cnn_3(x_3)
        x_3 = F.elu(self.enc_linear_3(x_3))
        code_3 = self.power_constraint(x_3)

        # channel of Phase 3.
        r_3_rec = code_3 + fwd_noise[:, :, 2].view(self.args.batch_size, self.args.block_len, 1)

        # quantize received signal
        if self.args.rec_quantize_level!=0:
            Channel_quantize = ChannelSTEQuantize.apply
            r_3 = Channel_quantize(r_3_rec, self.args.rec_quantize_limit, self.args.rec_quantize_level)
        else:
            r_3 = r_3_rec

        # FB Decoder
        received_codes = torch.cat([r_1, r_2, r_3], dim=2)

        return received_codes


class DeepCode_decoder(torch.nn.Module):
    def __init__(self, args):
        super(DeepCode_decoder, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        if args.cnn_type == 'dense':
            CNNModel = DenseSameShapeConv1d
        else:
            CNNModel = SameShapeConv1d

        # Code rate 1/3.
        if args.codec == 'deepcode_cnn':
            self.dec_cnn    = CNNModel(num_layer=args.dec_num_layer, in_channels=args.code_rate_n,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size)
            self.dec_output = torch.nn.Linear(args.dec_num_unit, args.code_rate_k)


        else:
            self.dec_rnn    = torch.nn.GRU(3,  args.dec_num_unit,
                                           num_layers=args.dec_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)
            self.dec_output = torch.nn.Linear(2*args.dec_num_unit, args.code_rate_k)

    def forward(self, received_codes):
        if self.args.codec == 'deepcode_cnn':
            x_dec     = self.dec_cnn(received_codes)
        else:
            x_dec, _  = self.dec_rnn(received_codes)
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

        self.fwd_enc1 = FTAE_encoder(args, p_array)
        self.fwd_enc2 = FTAE_encoder(args, p_array)
        self.fwd_enc3 = FTAE_encoder(args, p_array)

        self.fb_enc1 = FTAE_active_fb(args, p_array)
        self.fb_enc2 = FTAE_active_fb(args, p_array)

        self.dec = FTAE_encoder(args, p_array)

        # Now I only play with block delayed schemes.
        if args.codec in ['turboae_blockdelay_cnn','turboae_blockdelay_rnn']:
            self.enc = FTAE_encoder(args, p_array)
            self.dec = FTAE_decoder(args, p_array)

        elif args.codec in ['deepcode_cnn', 'deepcode_rnn']:
            self.enc = DeepCode_encoder(args)
            self.dec = DeepCode_decoder(args)

    def forward(self, inputs, fwd_h, fwd_z, fb_h, fb_z):
        # Decouple feedbacks
        fwd_h1  = fwd_h[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        fwd_h2  = fwd_h[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        fwd_h3  = fwd_h[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        fwd_z1  = fwd_z[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        fwd_z2  = fwd_z[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        fwd_z3  = fwd_z[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        fb_h1  = fb_h[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        fb_h2  = fb_h[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        fb_h3  = fb_h[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        fb_z1  = fb_z[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        fb_z2  = fb_z[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        fb_z3  = fb_z[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        # Forward transmission

        # Code Phase 1
        x_1 = self.fwd_enc1(inputs)
        y_1 = fading_channel(x_1, fwd_h1, fwd_z1)
        f_1 = self.fb_enc1(y_1)
        r_1 = fading_channel(f_1, fb_h1, fb_z1)

        # Code Phase 2
        if self.args.ignore_feedback:
            r_1 = r_1 * 0.0
        if self.args.ignore_prev_code:
            x_1 = x_1 * 0.0

        input_2 = torch.cat([inputs, x_1, r_1])
        x_2 = self.fwd_enc2(input_2)
        y_2 = fading_channel(x_2, fwd_h2, fwd_z2)
        f_2 = self.fb_enc1(y_2)
        r_2 = fading_channel(f_2, fb_h2, fb_z2)

        # Code Phase 3









        received_codes = self.enc(inputs, fwd_noise, feedback_noise)
        final = self.dec(received_codes)
        return final
