__author__ = 'yihanjiang'
# update 04/25/2019. Target to publish the code to NeurIPS for Deep Turbo Autoencoder (DTA)
# Use Python3
# Only contain Turbo Related Components. Discard all non-Turbo things

import torch
import torch.optim as optim
import numpy as np
import sys
from get_interpret_args import get_args
from trainer import train, validate, test

from numpy import arange
from numpy.random import mtrand


def import_enc(args):
    # choose encoder

    if args.encoder == 'dta_rate3_rnn':
        from encoders import ENC_interRNN as ENC

    elif args.encoder == 'dta_rate3_cnn':
        from encoders import ENC_interCNN as ENC

    elif args.encoder == 'dta_rate3_cnn_2inter':
        from encoders import ENC_interCNN2Int as ENC

    elif args.encoder == 'dta_rate3_cnn2d':
        from encoders import ENC_interCNN2D as ENC

    elif args.encoder == 'dta_rate3_rnn_sys':
        from encoders import ENC_interRNN_sys as ENC

    elif args.encoder == 'dta_rate2_rnn':
        from encoders import ENC_turbofy_rate2 as ENC

    elif args.encoder == 'dta_rate2_cnn':
        from encoders import ENC_turbofy_rate2_CNN as ENC  # not done yet

    else:
        print('Unknown Encoder, stop')

    return ENC

def import_dec(args):

    if args.decoder == 'dta_rate2_rnn':
        from decoders import DEC_LargeRNN_rate2 as DEC

    elif args.decoder == 'dta_rate2_cnn':
        from decoders import DEC_LargeCNN_rate2 as DEC  # not done yet

    elif args.decoder == 'dta_rate3_cnn':
        from decoders import DEC_LargeCNN as DEC

    elif args.decoder == 'dta_rate3_cnn_2inter':
        from decoders import DEC_LargeCNN as DEC

    elif args.decoder == 'dta_rate3_cnn2d':
        from decoders import DEC_LargeCNN2D as DEC

    elif args.decoder == 'dta_rate3_rnn':
        from decoders import DEC_LargeRNN as DEC

    elif args.decoder == 'nbcjr_rate3':                # ICLR 2018 paper
        from decoders import NeuralTurbofyDec as DEC

    return DEC

if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)


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

    # setup interleaver. very important for checking encoder
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


    encoder = ENC(args, p_array)
    decoder = DEC(args, p_array)

    # choose support channels
    from channel_ae import Channel_AE
    model = Channel_AE(args, encoder, decoder).to(device)

    # make the model parallel
    if args.is_parallel == 1:
        model.enc.set_parallel()
        model.dec.set_parallel()

    # weight loading
    if args.init_nw_weight == 'default':
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight, map_location='cpu')

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict = False)

        except:
            model.load_state_dict(pretrained_model, strict = False)

        model.args = args

    print(model)

    general_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)


    #################################################
    # Check Encoder
    #################################################


    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    flip_pos = 20
    general_optimizer.zero_grad()

    X_train_1    = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
    X_train_0    = X_train_1.clone()
    X_train_1[:, flip_pos, :] = 1.0
    X_train_0[:, flip_pos, :] = 0.0


    codes_1 = model.enc(X_train_1)
    codes_0 = model.enc(X_train_0)

    code_diff = torch.abs(codes_0 - codes_1)
    xx, _  = torch.max(code_diff, dim=0)
    print(xx.shape)
    code_diff = xx.detach().numpy()

    # # MI estimations.
    # data_shape = codes_1.shape
    # from trainer import generate_noise
    # fwd_noise = generate_noise(data_shape, args, snr_low=5.0, snr_high=5.0, mode = 'encoder')
    # #  Rayleigh Fading Channel, non-coherent
    # fading_h = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) #np.sqrt(2.0)
    # fading_h = fading_h.type(torch.FloatTensor).to(device)
    # received_codes = fading_h*codes_1 + fwd_noise
    #
    #
    # received_codes = received_codes.detach().cpu().numpy()
    # codes_1 = codes_1.detach().cpu().numpy()
    #
    # Y = received_codes.reshape(received_codes.shape[0]*received_codes.shape[1]*received_codes.shape[2],1 )
    # X = codes_1.reshape(codes_1.shape[0]*codes_1.shape[1]*codes_1.shape[2],1 )
    #
    #
    # from knnie.knnie import kraskov_mi
    #
    # print(np.std(Y), np.mean(Y))
    # print(np.std(X), np.mean(X))
    #
    # print("I(X;Y) = ", kraskov_mi(X,Y)/np.log(2.0))
    # print('SNR', 0.0)
    #
    #

    print([float(item) for item in code_diff[:, 0]])
    print([float(item) for item in code_diff[:, 1]])
    print([float(item) for item in code_diff[:, 2]])

    import matplotlib.pylab as plt

    plt.figure(1)


    p1, = plt.plot(code_diff[:, 0], label = 'enc1')
    p2, = plt.plot(code_diff[:, 1], label = 'enc2')
    p3, = plt.plot(code_diff[:, 2], label = 'enc3, interleaved')

    plt.legend(handles = [p1, p2, p3])

    plt.grid()
    plt.show()













