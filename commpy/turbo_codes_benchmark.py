__author__ = 'yihanjiang'
'''
Evaluate
'''
from utils import  corrupt_signal, snr_db2sigma, get_test_sigmas

import numpy as np
import time
import sys

import channelcoding.convcode as cc
import channelcoding.interleavers as RandInterlv

from utilities import hamming_dist
import channelcoding.turbo as turbo


import multiprocessing as mp


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=100)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-num_dec_iteration', type=int, default=6)

    parser.add_argument('-code_rate',  type=int, default=3)

    parser.add_argument('-M',  type=int, default=2)
    parser.add_argument('-enc1',  type=int, default=7)     #,7,5,7 is the best
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)

    parser.add_argument('-num_cpu', type=int, default=2)

    parser.add_argument('-snr_test_start', type=float, default=-1.5)
    parser.add_argument('-snr_test_end', type=float, default=2.0)
    parser.add_argument('-snr_points', type=int, default=8)

    parser.add_argument('-noise_type',        choices = ['awgn', 't-dist','radar', 'fading','bsc', 'bec',
                                                         'radar_saturate', 'radar_erasure',
                                                         'ge',
                                                         'ge_awgn'], default='awgn')

    parser.add_argument('-dec_type', choices = ['awgn', 't3', 't5', 'fading'], default='awgn')


    parser.add_argument('-radar_power',       type=float, default=20.0)
    parser.add_argument('-radar_prob',        type=float, default=0.05)
    parser.add_argument('-radar_denoise_thd', type=float, default=10.0)
    parser.add_argument('-v',                 type=float,   default=5.0)

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()
    print args
    print '[ID]', args.id
    return args



if __name__ == '__main__':
    args = get_args()

    M = np.array([args.M])                                       # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[args.enc1, args.enc2]])   # Encoder of convolutional encoder
    feedback = args.feedback                                # Feedback of convolutional encoder

    print '[testing] Turbo Code Encoder: G: ', generator_matrix,'Feedback: ', feedback,  'M: ', M
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)
    interleaver = RandInterlv.RandInterlv(args.block_len, 0)
    p_array = interleaver.p_array
    codec  = [trellis1, trellis2, interleaver]

    if args.noise_type not in ['bsc', 'bec', 'ge']:
        snrs, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)
    else:
        tmp = [args.snr_test_start + item * (args.snr_test_end -args.snr_test_start)/(args.snr_points-1)for item in range(args.snr_points)]
        snrs, test_sigmas = tmp, np.array(tmp)

    turbo_res_ber, turbo_res_bler= [], []

    tic = time.time()

    def turbo_compute((idx, x)):
        '''
        Compute Turbo Decoding in 1 iterations for one SNR point.
        '''
        np.random.seed()
        message_bits = np.random.randint(0, 2, args.block_len)
        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        sys_r  = corrupt_signal(sys, noise_type =args.noise_type, sigma = test_sigmas[idx],
                                   vv =args.v, radar_power = args.radar_power, radar_prob = args.radar_prob,
                                   denoise_thd = args.radar_denoise_thd)
        par1_r = corrupt_signal(par1, noise_type =args.noise_type, sigma = test_sigmas[idx],
                                   vv =args.v, radar_power = args.radar_power, radar_prob = args.radar_prob,
                                   denoise_thd = args.radar_denoise_thd)
        par2_r = corrupt_signal(par2, noise_type =args.noise_type, sigma = test_sigmas[idx],
                                   vv =args.v, radar_power = args.radar_power, radar_prob = args.radar_prob,
                                   denoise_thd = args.radar_denoise_thd)



        if args.noise_type not in ['bsc', 'bec', 'ge']:
            decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1,
                                                     test_sigmas[idx]**2, args.num_dec_iteration , interleaver, L_int = None)
        elif args.noise_type == 'fading':
            decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1,
                                                     test_sigmas[idx]**2, args.num_dec_iteration , interleaver, L_int = None, fading=True)
        else:
            decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1,
                                                     test_sigmas[idx]**2, args.num_dec_iteration , interleaver, L_int = None)

        num_bit_errors = hamming_dist(message_bits, decoded_bits)

        return num_bit_errors


    commpy_res_ber = []
    commpy_res_bler= []

    nb_errors = np.zeros(test_sigmas.shape)
    map_nb_errors = np.zeros(test_sigmas.shape)
    nb_block_no_errors = np.zeros(test_sigmas.shape)

    for idx in xrange(len(test_sigmas)):
        start_time = time.time()
        pool = mp.Pool(processes=args.num_cpu)
        results = pool.map(turbo_compute, [(idx,x) for x in xrange(int(args.num_block))])

        for result in results:
            if result == 0:
                nb_block_no_errors[idx] = nb_block_no_errors[idx]+1

        nb_errors[idx]+= sum(results)
        print '[testing]SNR: ' , snrs[idx]
        print '[testing]BER: ', sum(results)/float(args.block_len*args.num_block)
        print '[testing]BLER: ', 1.0 - nb_block_no_errors[idx]/args.num_block
        commpy_res_ber.append(sum(results)/float(args.block_len*args.num_block))
        commpy_res_bler.append(1.0 - nb_block_no_errors[idx]/args.num_block)
        end_time = time.time()
        print '[testing] This SNR runnig time is', str(end_time-start_time)


    print '[Result]SNR: ', snrs
    print '[Result]BER', commpy_res_ber
    print '[Result]BLER', commpy_res_bler

    toc = time.time()
    print '[Result]Total Running time:', toc-tic
