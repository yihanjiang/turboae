__author__ = 'yihanjiang'

import torch
from utils import snr_db2sigma, snr_sigma2db
import numpy as np

def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0, mode = 'encoder'):
    # SNRs at training
    if test_sigma == 'default':
        if args.channel == 'bec':
            if mode == 'encoder':
                this_sigma = args.bec_p_enc
            else:
                this_sigma = args.bec_p_dec

        elif args.channel in ['bsc', 'ge']:
            if mode == 'encoder':
                this_sigma = args.bsc_p_enc
            else:
                this_sigma = args.bsc_p_dec
        else: # general AWGN cases
            this_sigma_low = snr_db2sigma(snr_low)
            this_sigma_high= snr_db2sigma(snr_high)
            # mixture of noise sigma.
            this_sigma = (this_sigma_low - this_sigma_high) * torch.rand(noise_shape) + this_sigma_high

    else:
        if args.channel in ['bec', 'bsc', 'ge']:  # bsc/bec noises
            this_sigma = test_sigma
        else:
            this_sigma = snr_db2sigma(test_sigma)

    # SNRs at testing
    if args.channel == 'awgn':
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 't-dist':
        fwd_noise  = this_sigma * torch.from_numpy(np.sqrt((args.vv-2)/args.vv) * np.random.standard_t(args.vv, size = noise_shape)).type(torch.FloatTensor)

    elif args.channel == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], noise_shape,
                                       p=[1 - args.radar_prob, args.radar_prob])

        corrupted_signal = args.radar_power* np.random.standard_normal( size = noise_shape ) * add_pos
        fwd_noise = this_sigma * torch.randn(noise_shape, dtype=torch.float) +\
                    torch.from_numpy(corrupted_signal).type(torch.FloatTensor)

    elif args.channel == 'bec':
        fwd_noise = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape,
                                        p=[this_sigma, 1 - this_sigma])).type(torch.FloatTensor)

    elif args.channel == 'bsc':
        fwd_noise = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape,
                                        p=[this_sigma, 1 - this_sigma])).type(torch.FloatTensor)
    elif args.channel == 'ge_awgn':
        #G-E AWGN channel
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = snr_db2sigma(snr_sigma2db(this_sigma) + 1)          # accuracy on good state
        bsc_h = snr_db2sigma(snr_sigma2db(this_sigma) - 1)   # accuracy on good state

        fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        if test_sigma == 'default':
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_k[batch_idx,time_idx, code_idx]
                        else:
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_k
                        good = np.random.random()<p_gg
                    elif not good:
                        if test_sigma == 'default':
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_h[batch_idx,time_idx, code_idx]
                        else:
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_h
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)* torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 'ge':
        #G-E discrete channel
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = 1.0        # accuracy on good state
        bsc_h = this_sigma# accuracy on good state

        fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        tmp = np.random.choice([0.0, 1.0], p=[1-bsc_k, bsc_k])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_gg
                    elif not good:
                        tmp = np.random.choice([0.0, 1.0], p=[ 1-bsc_h, bsc_h])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)

    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise



