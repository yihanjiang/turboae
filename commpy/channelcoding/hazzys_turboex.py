import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding
import commpy.channelcoding.turbo as turbo
import commpy.channelcoding.interleavers as RandInterlv
import time
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from commpy.utilities import *

NType = 'iid'


# Block Length: try 1000, 10K, 50K, 100, 50
k = 1000 # Block Length.
print('k: ',k)


iterations_number = 10000000/k# Number of blocks for testing # (iter_num=1000, k = 1000, SNR_points = 10: 3 hours)
print('Number of Blocks: ', iterations_number)


# Testing sigmas
'''
if k == 50:
    test_sigmas = np.array([1.25892544,  1.12118053,  0.98343557,  0.84569067,  0.70794576], dtype='float32')
elif k == 100:
    test_sigmas = np.array([1.25892544,  1.12118053,  0.98343557,  0.84569067,  0.70794576], dtype='float32')
elif k == 1000:
    test_sigmas = np.array([1.25892544,  1.12118053 ,   1.0471 ,  0.9772 ,   0.9120 ,    0.84569067,   0.7943 ,  0.70794576,   0.6310], dtype = 'float32')
    #test_sigmas = np.array([1.25892544, 1.12118053, 1.07, 1.02, 0.98343557, 0.96, 0.92, 0.89, 0.87, 0.8, 0.64], dtype='float32')
elif k == 10000:
    test_sigmas = np.array([1.25892544, 1.12118053, 1.07, 1.02, 0.98343557, 0.96], dtype ='float32')
elif k == 50000:
    test_sigmas = np.array([1.25892544, 1.12118053, 1.07, 1.02, 0.98343557], dtype ='float32')
else:
    print('Unknown uumber of blocks.')
'''

SNR_dB_start_Eb = -1
SNR_dB_stop_Eb = 2
SNR_points = 10

# Testing sigmas for Direct link, Tx to Relay, Relay to Rx.
SNRS = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points, dtype = 'float32')
#SNRS = -10*np.log10(test_sigmas**2)
test_sigmas = 10**(-SNRS*1.0/20)

print SNRS
# =============================================================================
# Example showing the encoding and decoding of convolutional codes
# =============================================================================

# G(D) corresponding to the convolutional encoder

# Create trellis data structure
#trellis1 = cc.Trellis(np.array([3]), np.array([[11,13]]),feedback=11)
#trellis2 = cc.Trellis(np.array([3]), np.array([[11,13]]),feedback=11)
#print('trellis: cc.Trellis(np.array([3]), np.array([[11,13]]),feedback=11) ')

trellis1 = cc.Trellis(np.array([2]), np.array([[7,5]]),feedback=7)
trellis2 = cc.Trellis(np.array([2]), np.array([[7,5]]),feedback=7)
print('trellis: cc.Trellis(np.array([2]), np.array([[7,5]]),feedback=7) ')


interleaver = RandInterlv.RandInterlv(k,0)

nb_errors = np.zeros(test_sigmas.shape)
map_nb_errors = np.zeros(test_sigmas.shape)
nb_block_no_errors = np.zeros(test_sigmas.shape)


tic = time.clock()

for iterations in xrange(iterations_number):
    print(iterations)
    message_bits = np.random.randint(0, 2, k)
    [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

    for idx in xrange(len(test_sigmas)):
                        
        if NType == 'iid':
            noise = test_sigmas[idx]*np.random.standard_normal(sys.shape) # Define noise
            sys_r = (2*sys-1) + noise # Modulation plus noise
            noise = test_sigmas[idx]*np.random.standard_normal(par1.shape) # Define noise
            par1_r = (2*par1-1) + noise # Modulation plus noise
            noise = test_sigmas[idx]*np.random.standard_normal(par2.shape) # Define noise
            par2_r = (2*par2-1) + noise # Modulation plus noise

        decoded_bits = hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, test_sigmas[idx]**2, 6, interleaver, L_int = None)
        #decoded_bits = hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, test_sigmas[idx]**2, 6, interleaver, L_int = None)

        num_bit_errors = hamming_dist(message_bits, decoded_bits)

        nb_errors[idx]+= num_bit_errors
        # print(num_bit_errors)
        if num_bit_errors == 0:
            nb_block_no_errors[idx] = nb_block_no_errors[idx]+1

            

print('SNR: ')
print(SNRS)
print('Turbo decoder BER: ')
BERS = nb_errors/float(k*iterations_number)
print(BERS)   
print('BLER: ')
BLERS = 1-nb_block_no_errors/float(iterations_number)
print(BLERS)


toc = time.clock()

print('time:', toc-tic)


np.savetxt('hazzys_turboex_BL1000_BN10000_SNRS.txt', SNRS)

np.savetxt('hazzys_turboex_BL1000_BN10000_BERS.txt', BERS)

np.savetxt('hazzys_turboex_BL1000_BN10000_BLERS.txt', BLERS)

'''
legend = []

plt.plot(10*np.log10(1/(test_sigmas**2).astype(float)), nb_errors/float(k*iterations_number))
legend.append('Turbo BER') 

plt.legend(legend, loc=3)

plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')    
plt.grid(True)
#plt.show()
plt.savefig('Turbo_rate_1_3.png')

'''
##file = open('depth_'+str(tb_depth)+'_Gaussian_iid_Viterbi_ber.txt',"w")
##file.write( nb_errors/float(k*iterations_number) )
##
##file2 = open('depth_'+str(tb_depth)+'_Gaussian_iid_Viterbi_snr.txt',"w")
##file2.write(str(SNRS))
##          
##file.close()
##file2.close()
