__author__ = 'yihanjiang'
import torch
import torch.optim as optim
import numpy as np
import sys
from ftae_get_args import get_args
from ftae_trainer  import ftae_train, ftae_validate, ftae_test

from numpy import arange
from numpy.random import mtrand

from ftae_ae import Channel_Feedback_rate3


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

if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    # put all printed things to log file
    logfile = open('./logs/ftae'+identity+'_log.txt', 'a')
    sys.stdout = Logger('./logs/ftae'+identity+'_log.txt', sys.stdout)

    args = get_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # setup interleaver.

    if args.is_interleave == 1:
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array = rand_gen.permutation(arange(args.block_len))
        print('using random interleaver', p_array)

    elif args.is_interleave == 0:
        p_array = range(args.block_len)
    else:
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array = rand_gen.permutation(arange(args.block_len))
        print('using random interleaver', p_array)

    model = Channel_Feedback_rate3(args, p_array).to(device)

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
    enc_params = list(model.fwd_enc1.parameters()) + list(model.fwd_enc2.parameters()) + list(model.fwd_enc3.parameters())
    enc_optimizer = optim.Adam(enc_params,lr=args.enc_lr)

    dec_params = list(model.fb_enc1.parameters()) + list(model.fb_enc2.parameters()) + list(model.dec.parameters())
    dec_optimizer = optim.Adam(dec_params, lr=args.dec_lr)
    general_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################

    report_loss, report_ber, report_MI = [], [], []

    for epoch in range(1, args.num_epoch + 1):
        if args.num_train_enc > 0:
            for idx in range(args.num_train_enc):
                ftae_train(epoch, model, enc_optimizer, args, use_cuda = use_cuda, mode ='encoder')


        if args.num_train_dec > 0:
            for idx in range(args.num_train_dec):
                ftae_train(epoch, model, dec_optimizer, args, use_cuda = use_cuda, mode ='decoder')

        this_loss, this_ber = ftae_validate(model, general_optimizer, args, use_cuda = use_cuda)
        report_loss.append(this_loss)
        report_ber.append(this_ber)

    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################
    ftae_test(model, args, use_cuda = use_cuda)

    torch.save(model.state_dict(), './tmp/torch_model_'+identity+'.pt')
    print('saved model', './tmp/torch_model_'+identity+'.pt')




