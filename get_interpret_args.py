__author__ = 'yihanjiang'
import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-encoder', choices=['dta_rate3_rnn',      # DTA Encoder, rate 1/3, RNN
                                             'dta_rate3_rnn_sys',  # DTA Encoder, rate 1/3, Systematic Bit hard coded
                                             'dta_rate3_cnn',      # DTA Encoder, rate 1/3, Same Shape 1D CNN encoded.
                                             'dta_rate3_cnn2d',    # DTA Encoder, rate 1/3, Same Shape 2D CNN encoded.
                                             'dta_rate2_rnn',      # DTA Encoder, rate 1/2, RNN
                                             'dta_rate2_cnn',      # DTA Encoder, rate 1/2, Same Shape 1D CNN encoded.(TBD)
                                             'rate3_cnn'           # CNN Encoder, rate 1/3. No Interleaver
                                            ],
                        default='dta_rate3_cnn')

    parser.add_argument('-decoder', choices=['dta_rate3_rnn',      # DTA Decoder, rate 1/3
                                             'dta_rate3_cnn',      # DTA Decoder, rate 1/3, Same Shape 1D CNN Decoder
                                             'dta_rate3_cnn2d',    # DTA Encoder, rate 1/3, Same Shape 2D CNN encoded.
                                             'dta_rate2_rnn',      # DTA Decoder, rate 1/2
                                             'dta_rate2_cnn',      # DTA Decoder, rate 1/2
                                             'nbcjr_rate3',        # NeuralBCJR Decoder, rate 1/3, allow ft size.
                                             'rate3_cnn'           # CNN Encoder, rate 1/3. No Interleaver
                                            ],
                        default='dta_rate3_cnn')
    ################################################################
    # Channel related parameters
    ################################################################
    parser.add_argument('-channel', choices = ['awgn',             # AWGN
                                               't-dist',           # Non-AWGN, ATN, with -vv associated
                                               'radar',            # Non-AWGN, Radar, with -radar_prob, radar_power, associated
                                               'ge_awgn',          # Non-IID, stateful channel.
                                               'bec', 'bsc',       # Binary Channels
                                               'ge',               # Binary Non-IID stateful Channel
                                               'fading'            # Non-coherent Rayleigh Fading AWGN Channel
                                               ],
                        default = 'awgn')
    # Channel parameters
    parser.add_argument('-vv',type=float, default=5, help ='only for t distribution channel')

    parser.add_argument('-radar_prob',type=float, default=0.05, help ='only for radar distribution channel')
    parser.add_argument('-radar_power',type=float, default=5.0, help ='only for radar distribution channel')

    # binary channels training algorithms (not tested yet~)
    parser.add_argument('-bec_p',type=float, default=0.0, help ='only for bec channel, enc')
    parser.add_argument('-bsc_p',type=float, default=0.0, help ='only for bsc channel, enc')
    parser.add_argument('-bec_p_dec',type=float, default=0.0, help ='only for bec channel, dec')
    parser.add_argument('-bsc_p_dec',type=float, default=0.0, help ='only for bsc channel, dec')

    # continuous channels training algorithms
    parser.add_argument('-train_channel_low', type=float, default  = 2.0)
    parser.add_argument('-train_channel_high', type=float, default = 2.0)

    parser.add_argument('-train_dec_channel_low', type=float, default  = -1.5)
    parser.add_argument('-train_dec_channel_high', type=float, default = 2.0)


    parser.add_argument('-init_nw_weight', type=str, default='./models/dta_q2_atn_v3_trainenc2_traindec_neg15_2.pt')

    # code rate is k/n, so that enable multiple code rates.
    parser.add_argument('-code_rate_k', type=int, default=1)
    parser.add_argument('-code_rate_n', type=int, default=3)

    ################################################################
    # DTA encoder/decoder parameters
    ################################################################
    parser.add_argument('-enc_type', choices=['bd', 'sd'], default='bd') # not functional
    parser.add_argument('-dec_type', choices=['bd', 'sd'], default='bd') # not functional

    parser.add_argument('-enc_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-dec_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')

    parser.add_argument('-num_iteration', type=int, default=6)
    parser.add_argument('-extrinsic', type=int, default=1)
    parser.add_argument('-num_iter_ft', type=int, default=5)
    parser.add_argument('-is_interleave', type=int, default=1, help='0 is not interleaving, 1 is fixed interleaver, >1 is random interleaver')
    parser.add_argument('-is_same_interleaver', type=int, default=1, help='not random interleaver, potentially finetune?')
    parser.add_argument('-is_parallel', type=int, default=1)
    # CNN related
    parser.add_argument('-enc_kernel_size', type=int, default=5)
    parser.add_argument('-dec_kernel_size', type=int, default=5)

    # RNN related
    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=2)

    parser.add_argument('-enc_num_unit', type=int, default=100)
    parser.add_argument('-dec_num_unit', type=int, default=100)

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='elu')
    parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')


    ################################################################
    # STE related parameters
    ################################################################
    parser.add_argument('-enc_quantize_level', type=float, default=2, help = 'only valid for group_norm')
    parser.add_argument('-enc_value_limit', type=float, default=1.0, help = 'only valid for group_norm quantization')
    parser.add_argument('-enc_grad_limit', type=float, default=0.01, help = 'only valid for group_norm quantization')
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'default'], default='both', help = 'only valid for group_norm quantization')

    ################################################################
    # Training ALgorithm related parameters
    ################################################################
    parser.add_argument('-joint_train', type=int, default=0, help ='if 1, joint train enc+dec, 0: seperate train')
    parser.add_argument('-num_train_dec', type=int, default=1, help ='')
    parser.add_argument('-num_train_enc', type=int, default=1, help ='')

    parser.add_argument('-dropout',type=float, default=0.0)

    parser.add_argument('-snr_test_start', type=float, default=-1.5)
    parser.add_argument('-snr_test_end', type=float, default=4.0)
    parser.add_argument('-snr_points', type=int, default=12)

    parser.add_argument('-batch_size', type=int, default=1000)
    parser.add_argument('-num_epoch', type=int, default=1)
    parser.add_argument('-test_ratio', type=int, default=1,help = 'only for high SNR testing')
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-img_size', type=int, default=10, help='only used for CNN 2D structures')

    parser.add_argument('-num_block', type=int, default=1000)

    parser.add_argument('-test_channel_mode',
                        choices=['group_norm','group_norm_quantize'],
                        default='group_norm_quantize')
    parser.add_argument('-train_channel_mode',
                        choices=['group_norm','group_norm_quantize'],
                        default='group_norm_quantize')

    parser.add_argument('-group_norm_g', type = int, default=1)

    ################################################################
    # Optimizer related parameters
    ################################################################
    parser.add_argument('-optimizer', choices=['adam', 'momentum', 'rmsprop', 'adagrad'], default='adam')
    parser.add_argument('-dec_lr', type = float, default=0.001, help='decoder leanring rate')
    parser.add_argument('-enc_lr', type = float, default=0.001, help='encoder leanring rate')
    parser.add_argument('-momentum', type = float, default=0.9)

    parser.add_argument('-clip_norm', type = float, default=1.0)

    ################################################################
    # Loss related parameters
    ################################################################

    parser.add_argument('-loss', choices=['bce', 'mse','focal', 'bce_block', 'maxBCE'],
                        default='maxBCE', help='only BCE works')

    # focal loss related things
    parser.add_argument('-focal_gamma', type = float, default=0.0, help = 'default gamma=0,BCE; 5 is to emphasis wrongly decoded cases.')
    parser.add_argument('-focal_alpha', type = float, default=1.0, help = 'default alpha=1.0, adjust the loss term')

    # mixture term
    parser.add_argument('-lambda_maxBCE', type = float, default=0.01, help = 'add term to maxBCE loss, only wokr with maxBCE loss')

    ################################################################
    # MISC
    ################################################################
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--print_pos_ber', action='store_true', default=False,
                        help='print positional ber when testing BER')
    parser.add_argument('--print_pos_power', action='store_true', default=False,
                        help='print positional power when testing BER')
    parser.add_argument('--print_test_traj', action='store_true', default=False,
                        help='print positional power when testing BER')


    args = parser.parse_args()

    return args
