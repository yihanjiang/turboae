__author__ = 'yihanjiang'
# update 10/18/2019, code to replicate TurboAE paper in NeurIPS 2019.
# Tested on PyTorch 1.0.
# TBD: remove all non-TurboAE related functions.

import torch
import torch.optim as optim
import numpy as np
import sys
from get_args import get_args
from mod_trainer import train, validate, test

from numpy import arange
from numpy.random import mtrand

# utils for logger
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def import_enc(args):
    # choose encoder

    if args.encoder == 'TurboAE_rate3_rnn':
        from encoders import ENC_interRNN as ENC

    elif args.encoder in ['TurboAE_rate3_cnn', 'TurboAE_rate3_cnn_dense']:
        from encoders import ENC_interCNN as ENC

    elif args.encoder == 'turboae_2int':
        from encoders import ENC_interCNN2Int as ENC

    elif args.encoder == 'rate3_cnn':
        from encoders import CNN_encoder_rate3 as ENC

    elif args.encoder in ['TurboAE_rate3_cnn2d', 'TurboAE_rate3_cnn2d_dense']:
        from encoders import ENC_interCNN2D as ENC

    elif args.encoder == 'TurboAE_rate3_rnn_sys':
        from encoders import ENC_interRNN_sys as ENC

    elif args.encoder == 'TurboAE_rate2_rnn':
        from encoders import ENC_turbofy_rate2 as ENC

    elif args.encoder == 'TurboAE_rate2_cnn':
        from encoders import ENC_turbofy_rate2_CNN as ENC  # not done yet

    elif args.encoder in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
        from encoders import ENC_TurboCode as ENC          # DeepTurbo, encoder not trainable.

    elif args.encoder == 'rate3_cnn2d':
        from encoders import ENC_CNN2D as ENC

    else:
        print('Unknown Encoder, stop')

    return ENC

def import_dec(args):

    if args.decoder == 'TurboAE_rate2_rnn':
        from decoders import DEC_LargeRNN_rate2 as DEC

    elif args.decoder == 'TurboAE_rate2_cnn':
        from decoders import DEC_LargeCNN_rate2 as DEC  # not done yet

    elif args.decoder in ['TurboAE_rate3_cnn', 'TurboAE_rate3_cnn_dense']:
        from decoders import DEC_LargeCNN as DEC

    elif args.decoder == 'turboae_2int':
        from decoders import DEC_LargeCNN2Int as DEC

    elif args.encoder == 'rate3_cnn':
        from decoders import CNN_decoder_rate3 as DEC

    elif args.decoder in ['TurboAE_rate3_cnn2d', 'TurboAE_rate3_cnn2d_dense']:
        from decoders import DEC_LargeCNN2D as DEC

    elif args.decoder == 'TurboAE_rate3_rnn':
        from decoders import DEC_LargeRNN as DEC

    elif args.decoder == 'nbcjr_rate3':                # ICLR 2018 paper
        from decoders import NeuralTurbofyDec as DEC

    elif args.decoder == 'rate3_cnn2d':
        from decoders import DEC_CNN2D as DEC

    return DEC

if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    # put all printed things to log file
    logfile = open('./logs/'+identity+'_log.txt', 'a')
    sys.stdout = Logger('./logs/'+identity+'_log.txt', sys.stdout)

    args = get_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # choose encoder and decoder.
    ENC = import_enc(args)
    DEC = import_dec(args)

    # setup interleaver.
    if args.is_interleave == 1:           # fixed interleaver.
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
        p_array2 = rand_gen.permutation(arange(args.block_len))

    elif args.is_interleave == 0:
        p_array1 = range(args.block_len)   # no interleaver.
        p_array2 = range(args.block_len)   # no interleaver.
    else:
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array2 = rand_gen.permutation(arange(args.block_len))

    print('using random interleaver', p_array1, p_array2)

    if args.encoder == 'turboae_2int' and args.decoder == 'turboae_2int':
        encoder = ENC(args, p_array1, p_array2)
        decoder = DEC(args, p_array1, p_array2)
    else:
        encoder = ENC(args, p_array1)
        decoder = DEC(args, p_array1)

    # modulation and demodulations.
    from modulations import Modulation, DeModulation

    modulator = Modulation(args)
    demodulator = DeModulation(args)

    # choose support channels
    from channel_ae import Channel_ModAE
    model = Channel_ModAE(args, encoder, decoder, modulator, demodulator).to(device)



    # make the model parallel
    if args.is_parallel == 1:
        model.enc.set_parallel()
        model.dec.set_parallel()

    # weight loading
    if args.init_nw_weight == 'default':
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight)

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict = False)

        except:
            model.load_state_dict(pretrained_model, strict = False)

        model.args = args

    print(model)


    ##################################################################
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    if args.optimizer == 'lookahead':
        print('Using Lookahead Optimizers')
        from optimizers import Lookahead
        lookahead_k = 5
        lookahead_alpha = 0.5
        if args.num_train_enc != 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']: # no optimizer for encoder
            enc_base_opt  = optim.Adam(model.enc.parameters(), lr=args.enc_lr)
            enc_optimizer = Lookahead(enc_base_opt, k=lookahead_k, alpha=lookahead_alpha)

        if args.num_train_dec != 0:
            dec_base_opt  = optim.Adam(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)
            dec_optimizer = Lookahead(dec_base_opt, k=lookahead_k, alpha=lookahead_alpha)

        general_base_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)
        general_optimizer = Lookahead(general_base_opt, k=lookahead_k, alpha=lookahead_alpha)

    else: # Adam, SGD, etc....
        if args.optimizer == 'adam':
            OPT = optim.Adam
        elif args.optimizer == 'sgd':
            OPT = optim.SGD
        else:
            OPT = optim.Adam

        if args.num_train_enc != 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']: # no optimizer for encoder
            enc_optimizer = OPT(model.enc.parameters(),lr=args.enc_lr)

        if args.num_train_dec != 0:
            dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)

        if args.num_train_mod != 0:
            mod_optimizer = OPT(filter(lambda p: p.requires_grad, model.mod.parameters()), lr=args.mod_lr)

        if args.num_train_demod != 0:
            demod_optimizer = OPT(filter(lambda p: p.requires_grad, model.demod.parameters()), lr=args.demod_lr)

        general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber = [], []

    for epoch in range(1, args.num_epoch + 1):

        if args.joint_train == 1 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
            for idx in range(args.num_train_enc+args.num_train_dec):
                train(epoch, model, general_optimizer, args, use_cuda = use_cuda, mode ='encoder')

        else:
            if args.num_train_enc > 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
                for idx in range(args.num_train_enc):
                    train(epoch, model, enc_optimizer, args, use_cuda = use_cuda, mode ='encoder')

            if args.num_train_dec > 0:
                for idx in range(args.num_train_dec):
                    train(epoch, model, dec_optimizer, args, use_cuda = use_cuda, mode ='decoder')

            if args.num_train_mod > 0:
                for idx in range(args.num_train_mod):
                    train(epoch, model, mod_optimizer, args, use_cuda = use_cuda, mode ='decoder')

            if args.num_train_demod > 0:
                for idx in range(args.num_train_demod):
                    train(epoch, model, demod_optimizer, args, use_cuda = use_cuda, mode ='decoder')

        this_loss, this_ber  = validate(model, general_optimizer, args, use_cuda = use_cuda)
        report_loss.append(this_loss)
        report_ber.append(this_ber)

    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################

    torch.save(model.state_dict(), './tmp/torch_model_'+identity+'.pt')
    print('saved model', './tmp/torch_model_'+identity+'.pt')

    if args.is_variable_block_len:
        print('testing block length',args.block_len_low )
        test(model, args, block_len=args.block_len_low, use_cuda = use_cuda)
        print('testing block length',args.block_len )
        test(model, args, block_len=args.block_len, use_cuda = use_cuda)
        print('testing block length',args.block_len_high )
        test(model, args, block_len=args.block_len_high, use_cuda = use_cuda)

    else:
        test(model, args, use_cuda = use_cuda)














