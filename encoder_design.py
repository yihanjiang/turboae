__author__ = 'yihanjiang'
# update 06/01, try to see if encoder is enough to learn RSC code.

import torch
import torch.optim as optim
import numpy as np
import sys
from get_args import get_args
from trainer import train, validate, test

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

    if args.encoder == 'dta_rate3_rnn':
        from encoders import ENC_interRNN as ENC

    elif args.encoder == 'dta_rate3_cnn':
        from encoders import ENC_interCNN as ENC

    elif args.encoder == 'dta_rate3_cnn_2inter':
        from encoders import ENC_interCNN2Int as ENC

    elif args.encoder == 'rate3_cnn':
        from encoders import CNN_encoder_rate3 as ENC

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
        pretrained_model = torch.load(args.init_nw_weight)

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict = False)

        except:
            model.load_state_dict(pretrained_model, strict = False)

        model.args = args

    print(model)



    #################################################
    # Setup Optimizers
    #################################################
    if args.num_train_enc != 0:
        enc_optimizer = optim.Adam(model.enc.parameters(),lr=args.enc_lr)

    if args.num_train_dec != 0:
        dec_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)

    general_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################

    report_loss, report_ber, report_MI = [], [], []
    baseline    = 0.0

    for epoch in range(1, args.num_epoch + 1):

        if args.joint_train == 1:
            for idx in range(args.num_train_enc+args.num_train_dec):
                train(epoch, model, general_optimizer, args, use_cuda = use_cuda, mode ='encoder')

        else:
            if args.num_train_enc > 0:
                for idx in range(args.num_train_enc):
                    train(epoch, model, enc_optimizer, args, use_cuda = use_cuda, mode ='encoder')

            if args.num_train_dec > 0:
                for idx in range(args.num_train_dec):
                    train(epoch, model, dec_optimizer, args, use_cuda = use_cuda, mode ='decoder')

        this_loss, this_ber, this_MI = validate(model, general_optimizer, args, use_cuda = use_cuda)
        report_loss.append(this_loss)
        report_ber.append(this_ber)
        report_MI.append(this_MI)

    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        #print('test MI estimated', report_MI)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################
    test(model, args, use_cuda = use_cuda)

    torch.save(model.state_dict(), './tmp/torch_model_'+identity+'.pt')
    print('saved model', './tmp/torch_model_'+identity+'.pt')




