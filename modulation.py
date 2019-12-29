__author__ = 'yihanjiang'

'''
This requires to upgrade real value channels to complex channels, which requires quite a lot of engineering.
For now just use BPSK to make things easier.
BPSK is a over-loaded idea: means just real-value channel.

Modulation means each symbol has limited power.
Modulati
Input: continuous value
'''

def modulation(input_signal, mod_mode = 'bpsk'):
    if mod_mode == 'bpsk':
        return input_signal

    elif mod_mode == 'continuous_complex':
        # build block_len * code_rate / 2 symbols.
        pass

    elif mod_mode == 'qpsk':
        pass

    elif mod_mode == 'qam':
        pass



def demod(rec_signal, demod_mode = 'bpsk'):
    if demod_mode == 'bpsk':
        return rec_signal



