__author__ = 'yihanjiang'
'''
This is util functions of the decoder
SHall involve:
(1) Parallel Generating Turbo Code?
'''

import numpy as np
import math
import commpy.channelcoding.turbo as turbo

import scipy
import scipy.stats



#from scipy import stats
import keras.backend as K
import tensorflow as tf

#######################################
# Interleaving Helper Functions
#######################################
def deint(in_array, p_array):
    out_array = np.zeros(len(in_array), in_array.dtype)
    for index, element in enumerate(p_array):
        out_array[element] = in_array[index]
    return out_array

def intleave(in_array, p_array):
    out_array = np.array(map(lambda x: in_array[x], p_array))
    return out_array

def direct_subtract(in1,in2):
    out = in1
    out[:,:,2] = in1[:,:,2] - in2
    return out

#######################################
# Noise Helper Function
#######################################
'''
Use Corrupt_signal, generate_noise is depreciated.
'''
def corrupt_signal(input_signal, noise_type, sigma = 1.0,
                    vv =5.0, radar_power = 20.0, radar_prob = 5e-2, denoise_thd = 10.0,
                    modulate_mode = 'bpsk', snr_mixture = [0, 0, 0]):
    '''
    Documentation TBD.
    only bpsk is allowed, but other modulation is allowed by user-specified modulation.
    :param noise_type: required, choose from 'awgn', 't-dist'
    :param sigma:
    :param data_shape:
    :param vv: parameter for t-distribution.
    :param radar_power:
    :param radar_prob:
    :return:
    '''

    data_shape = input_signal.shape  # input_signal has to be a numpy array.

    if noise_type == 'awgn':
        noise = sigma * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'bsc':
        noise =  np.random.choice([-1.0, 1.0], data_shape, p=[sigma, 1 - sigma])
        corrupted_signal = (2.0*input_signal-1.0) * noise

    elif noise_type == 'bec':
        noise =  np.random.choice([0.0, 1.0], data_shape, p=[sigma, 1 - sigma])
        corrupted_signal = (2.0*input_signal-1.0) * noise

    elif noise_type == 'ge':
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = 1.0        # accuracy on good state
        bsc_h = sigma      # accuracy on good state
        fwd_noise = np.zeros(data_shape)

        good = True
        for time_idx in range(data_shape[0]):
            if good:
                tmp = np.random.choice([-1.0, 1.0], p=[1-bsc_k, bsc_k])
                fwd_noise[time_idx] = tmp
                good = np.random.random()<p_gg
            elif not good:
                tmp = np.random.choice([-1.0, 1.0], p=[1-bsc_h, bsc_h])
                fwd_noise[time_idx] = tmp
                good = np.random.random()<p_bb
            else:
                print('bad!!! something happens')

        noise =  fwd_noise
        corrupted_signal = (2.0*input_signal-1.0) * noise

    elif noise_type == 'ge_awgn':
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = snr_db2sigma(snr_sigma2db(sigma) + 1)        # accuracy on good state
        bsc_h = snr_db2sigma(snr_sigma2db(sigma) - 1)      # accuracy on good state

        fwd_noise = np.zeros(data_shape)

        good = True
        for time_idx in range(data_shape[0]):
            if good:
                tmp = bsc_k * np.random.standard_normal() # Define noise
                fwd_noise[time_idx] = tmp
                good = np.random.random()<p_gg
            elif not good:
                tmp = bsc_h * np.random.standard_normal() # Define noise
                fwd_noise[time_idx] = tmp
                good = np.random.random()<p_bb
            else:
                print('bad!!! something happens')



        noise =  fwd_noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'fading':

        fading_h = np.sqrt(np.random.standard_normal(data_shape)**2 +  np.random.standard_normal(data_shape)**2)/np.sqrt(3.14/2.0)
        noise = sigma * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = fading_h *(2.0*input_signal-1.0) + noise

    elif noise_type == 't-dist':
        noise = sigma * math.sqrt((vv-2.0)/vv) *np.random.standard_t(vv, size = data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'awgn+radar':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        add_poscomp = np.ones(data_shape) - abs(add_pos)

        corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

        # noise = sigma * np.random.standard_normal(data_shape) + \
        #         np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        #
        # corrupted_signal = 2.0*input_signal-1.0  + noise

    elif noise_type == 'awgn+radar+denoise':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        add_poscomp = np.ones(data_shape) - abs(add_pos)
        corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

        corrupted_signal  = np.clip(corrupted_signal, -denoise_thd, denoise_thd, out=0.0)

        # noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        # corrupted_signal = 2.0*input_signal-1.0 + noise
        # corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-denoise_thd, threshmax=denoise_thd, newval=0.0)

    elif noise_type == 'radar':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([0.0, 1.0], data_shape, p=[1 - radar_prob, radar_prob])
        corrupted_signal = bpsk_signal + radar_power* np.random.standard_normal( size = data_shape ) * add_pos

        #
        # burst = np.random.randint(0, data_shape[0], int(radar_prob*data_shape[0]))
        # bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        # corrupt_signal[burst] = bpsk_signal[burst] + radar_power * np.random.standard_normal(data_shape)


    elif noise_type == 'hyeji_bursty+denoise' or noise_type == 'radar_saturate' or noise_type == 'radar_erasure':

        def denoise_thd_func():
            sigma_1 = sigma
            sigma_2 = radar_power
            optimal_thd = math.sqrt( (2*(sigma_1**2)*(sigma_1**2 + sigma_2**2)/(sigma_2**2)) * math.log(math.sqrt(sigma_1**2 + sigma_2**2)/sigma_1))
            return optimal_thd

        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([0.0, 1.0], data_shape, p=[1 - radar_prob, radar_prob])
        corrupted_signal = bpsk_signal + radar_power* np.random.standard_normal( size = data_shape ) * add_pos

        #a = denoise_thd
        if denoise_thd == 10.0:
            a = denoise_thd_func() + 1
        else:
            a = denoise_thd

        if noise_type == 'hyeji_bursty+denoise' or noise_type == 'radar_erasure':

            corrupted_signal  = np.clip(corrupted_signal, -a, a, out=corrupted_signal)
            corrupted_signal = corrupted_signal * np.array(corrupted_signal != a, dtype=np.float)

        else:
            corrupted_signal  = np.clip(corrupted_signal, -a, a, out=corrupted_signal)

        return corrupted_signal

    elif noise_type == 'mixture-normalized':

        ref_snr = 0
        ref_sigma= 10**(-ref_snr*1.0/20)# reference is always 0dB.

        noise = sigma * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

        bpsk_signal_ref = 2.0*input_signal-1.0 + ref_sigma * np.random.standard_normal(data_shape)
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        pstate1 = 0.5
        add_pos     = np.random.choice([0, 1.0], data_shape, p=[pstate1,1-pstate1])
        add_poscomp = np.ones(data_shape) - abs(add_pos)

        corrupted_signal = bpsk_signal_ref * add_poscomp *1.0/(ref_sigma**2) + bpsk_signal * add_pos *1.0/(sigma**2)

        return corrupted_signal

    elif noise_type == 'mixture':

        ref_snr = 0
        ref_sigma= 10**(-ref_snr*1.0/20)# reference is always 0dB.

        noise = sigma * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

        bpsk_signal_ref = 2.0*input_signal-1.0 + ref_sigma * np.random.standard_normal(data_shape)
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        pstate1 = 0.5
        add_pos     = np.random.choice([0, 1.0], data_shape, p=[pstate1,1-pstate1])
        add_poscomp = np.ones(data_shape) - abs(add_pos)

        corrupted_signal = bpsk_signal_ref * add_poscomp *1.0 + bpsk_signal * add_pos *1.0

        return corrupted_signal

    elif noise_type == 'mix_snr_turbo':
        noise = snr_mixture[0] * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'random_snr_turbo':
        this_snr = np.random.uniform(snr_mixture[2],snr_mixture[0], data_shape)
        noise = np.multiply(this_snr, np.random.standard_normal(data_shape)) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    else:
        print '[Warning][Noise Generator]noise_type noty specified!'
        noise = sigma * np.random.standard_normal(data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + noise


    return corrupted_signal


# deprecated
def generate_noise(noise_type, sigma, data_shape, vv =5.0, radar_power = 20.0, radar_prob = 5e-2):
    '''
    Documentation TBD.
    :param noise_type: required, choose from 'awgn', 't-dist'
    :param sigma:
    :param data_shape:
    :param vv: parameter for t-distribution.
    :param radar_power:
    :param radar_prob:
    :return:
    '''
    if noise_type == 'awgn':
        noise = sigma * np.random.standard_normal(data_shape) # Define noise

    elif noise_type == 't-dist':
        noise = sigma * math.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)

    elif noise_type == 'awgn+radar':
        noise = sigma * np.random.standard_normal(data_shape) + \
                np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])

    elif noise_type == 'radar':
        noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])

    else:
        noise = sigma * np.random.standard_normal(data_shape)

    return noise

#######################################
# Build RNN Feed Helper Function (for Turbo Code only, need to refactor)
#######################################

def build_rnn_data_feed(num_block, block_len, noiser, codec, is_all_zero = False ,is_same_code = False, **kwargs):
    '''

    :param num_block:
    :param block_len:
    :param noiser: list, 0:noise_type, 1:sigma,     2:v for t-dist, 3:radar_power, 4:radar_prob
    :param codec:  list, 0:trellis1,   1:trellis2 , 2:interleaver
    :param kwargs:
    :return: X_feed, X_message
    '''

    # Unpack Noiser
    noise_type  = noiser[0]
    noise_sigma = noiser[1]
    vv          = 5.0
    radar_power = 20.0
    radar_prob  = 5e-2
    denoise_thd = 10.0
    snr_mix     = [0, 0, 0]

    if noise_type == 't-dist':
        vv = noiser[2]
    elif noise_type == 'awgn+radar' or noise_type == 'hyeji_bursty':
        radar_power = noiser[3]
        radar_prob  = noiser[4]

    elif noise_type == 'awgn+radar+denoise' or noise_type == 'hyeji_bursty+denoise':
        radar_power = noiser[3]
        radar_prob  = noiser[4]
        denoise_thd = noiser[5]

    elif noise_type == 'mix_snr_turbo' or noise_type == 'random_snr_turbo':
        snr_mix = noiser[6]

    elif noise_type == 'customize':
        '''
        TBD, noise model shall be open to other user, for them to train their own decoder.
        '''

        print '[Debug] Customize noise model not supported yet'
    else:  # awgn
        pass

    #print '[Build RNN Data] noise type is ', noise_type, ' noiser', noiser

    # Unpack Codec
    trellis1    = codec[0]
    trellis2    = codec[1]
    interleaver = codec[2]


    p_array     = interleaver.p_array

    X_feed = []
    X_message = []

    same_code = np.random.randint(0, 2, block_len)

    for nbb in range(num_block):
        if is_same_code:
            message_bits = same_code
        else:
            if is_all_zero == False:
                message_bits = np.random.randint(0, 2, block_len)
            else:
                message_bits = np.random.randint(0, 1, block_len)

        X_message.append(message_bits)
        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        sys_r  = corrupt_signal(sys, noise_type =noise_type, sigma = noise_sigma,
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd,
                               snr_mixture = snr_mix)
        par1_r = corrupt_signal(par1, noise_type =noise_type, sigma = noise_sigma,
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd,
                               snr_mixture = snr_mix)
        par2_r = corrupt_signal(par2, noise_type =noise_type, sigma = noise_sigma,
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd,
                               snr_mixture = snr_mix)

        rnn_feed_raw = np.stack([sys_r, par1_r, np.zeros(sys_r.shape), intleave(sys_r, p_array), par2_r], axis = 0).T
        rnn_feed = rnn_feed_raw

        X_feed.append(rnn_feed)

    X_feed = np.stack(X_feed, axis=0)

    X_message = np.array(X_message)
    X_message = X_message.reshape((-1,block_len, 1))

    return X_feed, X_message

#######################################
# Helper Function for convert SNR
#######################################

def snr_db2sigma(train_snr):
    block_len    = 1000
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr    = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    return sigma_snr

def snr_sigma2db(sigma_snr):
    SNR          = -10*np.log10(sigma_snr**2)
    return SNR

def get_test_sigmas(snr_start, snr_end, snr_points):
    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_end
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(1.0/2.0) for item in SNRS_dB]

    # my setup for SNR
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])
    #test_sigmas = np.array([(10**(-item*1.0/20))*math.sqrt(1.5) for item in SNRS_dB])

    # Turbo CRC SNR setup.
    test_sigmas = np.array([math.sqrt(0.5/((1.0/3)*(10.0**(0.1*item)))) for item in SNRS_dB])


    SNRS = SNRS_dB
    print '[testing] SNR range in dB ', SNRS

    return SNRS, test_sigmas

def code_err(y_true, y_pred):
    '''
    I want the 'sharpened' y_pred to be as close as BPSK codes.
    :param y_true:
    :param y_pred:
    :return:
    '''
    x = y_pred
    x = (100000.0 * x) + 0.0
    x = K.clip(x, 0.0, 1.0)
    x = 2*x - 1

    myOtherTensor = K.not_equal(y_true, K.round(x))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

def conv_enc(X_train_raw, args):

    import commpy.channelcoding.convcode as cc
    num_block = X_train_raw.shape[0]
    block_len = X_train_raw.shape[1]
    x_code    = []

    M = np.array([2]) # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[args.enc1, args.enc2]])

    feedback = args.feedback

    trellis = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure

    for idx in range(num_block):
        xx = cc.conv_encode(X_train_raw[idx, :, 0], trellis, 'rsc')

        xx1 = xx[::2]
        xx2 = xx[1::2]
        xx1 = xx1[:-int(M)]
        xx2 = xx2[:-int(M)]

        xx = np.array([xx1, xx2]).T

        # xx = xx[:-2*int(M)]
        # xx = xx.reshape((block_len, 2))

        x_code.append(xx)

    return np.array(x_code)





if __name__ == '__main__':
    pass