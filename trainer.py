__author__ = 'yihanjiang'
import torch
import time
import torch.nn.functional as F

eps  = 1e-6

from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler
from loss import customized_loss
from channels import generate_noise

import numpy as np
from numpy import arange
from numpy.random import mtrand

######################################################################################
#
# Trainer, validation, and test for AE code design
#
######################################################################################


def train(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    start_time = time.time()
    train_loss = 0.0
    k_same_code_counter = 0


    for batch_idx in range(int(args.num_block/args.batch_size)):


        if args.is_variable_block_len:
            block_len = np.random.randint(args.block_len_low, args.block_len_high)
        else:
            block_len = args.block_len

        optimizer.zero_grad()

        if args.is_k_same_code and mode == 'encoder':
            if batch_idx == 0:
                k_same_code_counter += 1
                X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
            elif k_same_code_counter == args.k_same_code:
                k_same_code_counter = 1
                X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
            else:
                k_same_code_counter += 1
        else:
            X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
        # train encoder/decoder with different SNR... seems to be a good practice.
        if mode == 'encoder':
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

        output, code = model(X_train, fwd_noise)
        output = torch.clamp(output, 0.0, 1.0)

        if mode == 'encoder':
            loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)

        else:
            loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)
            #loss = F.binary_cross_entropy(output, X_train)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss



def validate(model, optimizer, args, use_cuda = False, verbose = True):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss, test_custom_loss, test_ber= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
            fwd_noise  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)

            X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

            optimizer.zero_grad()
            output, codes = model(X_test, fwd_noise)

            output = torch.clamp(output, 0.0, 1.0)

            output = output.detach()
            X_test = X_test.detach()

            test_bce_loss += F.binary_cross_entropy(output, X_test)
            test_custom_loss += customized_loss(output, X_test, noise = fwd_noise, args = args, code = codes)
            test_ber  += errors_ber(output,X_test)


    test_bce_loss /= num_test_batch
    test_custom_loss /= num_test_batch
    test_ber  /= num_test_batch

    if verbose:
        print('====> Test set BCE loss', float(test_bce_loss),
              'Custom Loss',float(test_custom_loss),
              'with ber ', float(test_ber),
        )

    report_loss = float(test_bce_loss)
    report_ber  = float(test_ber)

    return report_loss, report_ber


def test(model, args, block_len = 'default',use_cuda = False):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    if block_len == 'default':
        block_len = args.block_len
    else:
        pass

    # Precomputes Norm Statistics.
    if args.precompute_norm_stats:
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size)* args.test_ratio)
            for batch_idx in range(num_test_batch):
                X_test = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
                X_test = X_test.to(device)
                _      = model.enc(X_test)
            print('Pre-computed norm statistics mean ',model.enc.mean_scalar, 'std ', model.enc.std_scalar)

    ber_res, bler_res = [], []
    ber_res_punc, bler_res_punc = [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber, test_bler = .0, .0
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test     = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
                fwd_noise  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

                X_hat_test, the_codes = model(X_test, fwd_noise)


                test_ber  += errors_ber(X_hat_test,X_test)
                test_bler += errors_bler(X_hat_test,X_test)

                if batch_idx == 0:
                    test_pos_ber = errors_ber_pos(X_hat_test,X_test)
                    codes_power  = code_power(the_codes)
                else:
                    test_pos_ber += errors_ber_pos(X_hat_test,X_test)
                    codes_power  += code_power(the_codes)

            if args.print_pos_power:
                print('code power', codes_power/num_test_batch)
            if args.print_pos_ber:
                res_pos = test_pos_ber/num_test_batch
                res_pos_arg = np.array(res_pos.cpu()).argsort()[::-1]
                res_pos_arg = res_pos_arg.tolist()
                print('positional ber', res_pos)
                print('positional argmax',res_pos_arg)
            try:
                test_ber_punc, test_bler_punc = .0, .0
                for batch_idx in range(num_test_batch):
                    X_test     = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
                    fwd_noise  = generate_noise(X_test.shape, args, test_sigma=sigma)
                    X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

                    X_hat_test, the_codes = model(X_test, fwd_noise)

                    test_ber_punc  += errors_ber(X_hat_test,X_test, positions = res_pos_arg[:args.num_ber_puncture])
                    test_bler_punc += errors_bler(X_hat_test,X_test, positions = res_pos_arg[:args.num_ber_puncture])

                    if batch_idx == 0:
                        test_pos_ber = errors_ber_pos(X_hat_test,X_test)
                        codes_power  = code_power(the_codes)
                    else:
                        test_pos_ber += errors_ber_pos(X_hat_test,X_test)
                        codes_power  += code_power(the_codes)
            except:
                print('no pos BER specified.')

        test_ber  /= num_test_batch
        test_bler /= num_test_batch
        print('Test SNR',this_snr ,'with ber ', float(test_ber), 'with bler', float(test_bler))
        ber_res.append(float(test_ber))
        bler_res.append( float(test_bler))

        try:
            test_ber_punc  /= num_test_batch
            test_bler_punc /= num_test_batch
            print('Punctured Test SNR',this_snr ,'with ber ', float(test_ber_punc), 'with bler', float(test_bler_punc))
            ber_res_punc.append(float(test_ber_punc))
            bler_res_punc.append( float(test_bler_punc))
        except:
            print('No puncturation is there.')

    print('final results on SNRs ', snrs)
    print('BER', ber_res)
    print('BLER', bler_res)
    print('final results on punctured SNRs ', snrs)
    print('BER', ber_res_punc)
    print('BLER', bler_res_punc)

    # compute adjusted SNR. (some quantization might make power!=1.0)
    enc_power = 0.0
    with torch.no_grad():
        for idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
            X_test     = X_test.to(device)
            X_code     = model.enc(X_test)
            enc_power +=  torch.std(X_code)
    enc_power /= float(num_test_batch)
    print('encoder power is',enc_power)
    adj_snrs = [snr_sigma2db(snr_db2sigma(item)/enc_power) for item in snrs]
    print('adjusted SNR should be',adj_snrs)

















