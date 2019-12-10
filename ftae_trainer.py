__author__ = 'yihanjiang'
import torch
import time
import torch.nn.functional as F

eps  = 1e-6

from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler
from loss import customized_loss
from channels import generate_noise

######################################################################################
#
# Trainer, validation, and test for Feedback Block Delay Channel Autoencoder
#
######################################################################################


def ftae_train(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    start_time = time.time()
    train_loss = 0.0

    for batch_idx in range(int(args.num_block/args.batch_size)):

        optimizer.zero_grad()
        X_train    = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

        if mode == 'encoder':
            fwd_noise  = generate_noise(X_train.shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(X_train.shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        fb_noise = generate_noise(X_train.shape, args, snr_low=args.fb_channel_low, snr_high=args.fb_channel_high, mode = 'decoder')

        X_train, fwd_noise, fb_noise = X_train.to(device), fwd_noise.to(device), fb_noise.to(device)

        output, code = model(X_train, fwd_noise, fb_noise)
        output = torch.clamp(output, 0.0, 1.0)

        if mode == 'encoder':
            loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)

        else:
            loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss

def ftae_validate(model, optimizer, args, use_cuda = False, verbose = True):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss, test_custom_loss, test_ber= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            fwd_noise  = generate_noise(X_test.shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)

            fb_noise = generate_noise(X_test.shape, args,
                                      snr_low=args.fb_channel_low, snr_high=args.fb_channel_high, mode = 'decoder')

            X_test, fwd_noise, fb_noise= X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

            optimizer.zero_grad()
            output, codes = model(X_test, fwd_noise, fb_noise)

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


def ftae_test(model, args, use_cuda = False):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    # Precomputes Norm Statistics.
    if args.precompute_norm_stats:
        num_test_batch = int(args.num_block/(args.batch_size)* args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test = X_test.to(device)
            _      = model.enc(X_test)
        print('Pre-computed norm statistics mean ',model.enc.mean_scalar, 'std ', model.enc.std_scalar)

    ber_res, bler_res = [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber, test_bler = .0, .0
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size)* args.test_ratio)
            for batch_idx in range(num_test_batch):
                X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                fwd_noise  = generate_noise(X_test.shape, args, test_sigma=sigma)

                fb_noise = generate_noise(X_test.shape, args,
                                          snr_low=args.fb_channel_low, snr_high=args.fb_channel_high, mode = 'decoder')

                X_test, fwd_noise, fb_noise= X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

                X_hat_test, the_codes = model(X_test, fwd_noise, fb_noise)


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
                print('positional ber', test_pos_ber/num_test_batch)

        test_ber  /= num_test_batch
        test_bler /= num_test_batch
        print('Test SNR',this_snr ,'with ber ', float(test_ber), 'with bler', float(test_bler))
        ber_res.append(float(test_ber))
        bler_res.append( float(test_bler))

    print('final results on SNRs ', snrs)
    print('BER', ber_res)
    print('BLER', bler_res)



















