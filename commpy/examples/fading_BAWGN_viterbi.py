import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# import convcode as cc
from commpy.utilities import *

NType = 'iid'
print(NType)
iterations_number = 100

k = 1000#00#0
N = 2*k#(k+2)#00#0

SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 10
SNR_points = 10

SNR_dB_start_Es = SNR_dB_start_Eb + 10*np.log10(float(k)/float(N))
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10*np.log10(float(k)/float(N))

sigma_start = np.sqrt(1/(2*10**(float(SNR_dB_start_Es)/float(10))))
sigma_stop = np.sqrt(1/(2*10**(float(SNR_dB_stop_Es)/float(10))))

# Testing sigmas
test_sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

SNRS = -10*np.log10(test_sigmas**2)


# =============================================================================
# Example showing the encoding and decoding of convolutional codes
# =============================================================================

# G(D) corresponding to the convolutional encoder
generator_matrix = np.array([[05, 07]])
#generator_matrix = np.array([[03, 00, 02], [07, 04, 06]])

# Number of delay elements in the convolutional encoder
M = np.array([2])

# Create trellis data structure
trellis = cc.Trellis(M, generator_matrix)

nb_errors = np.zeros(test_sigmas.shape)
map_nb_errors = np.zeros(test_sigmas.shape)


# Traceback depth of the decoder
tb_depth = 10#5*(M.sum() + 1)

print('traceback depth: '+str(tb_depth))

for idx in xrange(SNR_points):
    print(idx)
    for iterations in xrange(iterations_number):  

        message_bits = np.random.randint(0, 2, k)

        coded_bits = cc.conv_encode(message_bits, trellis)
#        print('coded_bits: ')
#        print(coded_bits)
        if NType == 'iid':
            noise = test_sigmas[idx]*np.random.standard_normal(coded_bits.shape) # Define noise

            uu = np.sqrt(0.5)*np.random.standard_normal(coded_bits.shape)
            vv = np.sqrt(0.5)*np.random.standard_normal(coded_bits.shape)
            rayleigh_coeff = np.sqrt(uu**2 + vv**2)                

            received = rayleigh_coeff*(2*coded_bits-1) + noise # Modulation plus noise

#        print('received: ')
#        print(received)

        decoded_bits = cc.viterbi_decode(received.astype(float), trellis, tb_depth,decoding_type='unquantized')

        received_odd = received[::2] # first, third, fifth, etc
        received_even = received[1::2] # second, fourth, sixth, etc 
        _, map_decoded_bits = commpy.channelcoding.map_decode(sys_symbols=received_odd.astype(float),  non_sys_symbols=received_even.astype(float), trellis = trellis, noise_variance = test_sigmas[idx]**2, L_int = np.zeros(received_odd.shape), mode = 'decode')
#        print('decoded_bits: ')
#        print(decoded_bits)

        num_bit_errors = hamming_dist(message_bits, decoded_bits[:-int(M)])

        map_num_bit_errors = hamming_dist(message_bits, map_decoded_bits[:-int(M)])

        nb_errors[idx]+= num_bit_errors
        map_nb_errors[idx]+= map_num_bit_errors

    print('SNR: ' + str(SNRS[idx]))
    print('Viterbi BER: '+ str(nb_errors[idx]/float(k*iterations_number)))   
    print('MAP(BCJR) BER: '+ str(map_nb_errors[idx]/float(k*iterations_number))+'\n')


legend = []

plt.plot(10*np.log10(1/(test_sigmas**2).astype(float)), nb_errors/float(k*iterations_number))
legend.append('Viterbi') 

plt.plot(10*np.log10(1/(test_sigmas**2).astype(float)), map_nb_errors/float(k*iterations_number))
legend.append('MAP(BCJR)') 

plt.legend(legend, loc=2)

plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')    
plt.grid(True)
plt.savefig('Fading_Viterbi_and_MAP_'+str(NType)+'.png')

##
##file = open('depth_'+str(tb_depth)+'_Gaussian_iid_Viterbi_ber.txt',"w")
##file.write( nb_errors/float(k*iterations_number) )
##
##file2 = open('depth_'+str(tb_depth)+'_Gaussian_iid_Viterbi_snr.txt',"w")
##file2.write(str(SNRS))
##          
##file.close()
##file2.close()
