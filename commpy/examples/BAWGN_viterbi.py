import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# import convcode as cc
from commpy.utilities import *

# NType = 'periodic'
NType = 'iid'
print ('NType:', NType)

print(NType)
iterations_number = 100
print 'iterations_number: ', iterations_number

k = 1000#0#000#00#0
print 'k: ',k
N = 2*(k+2)#00#0

SNR_dB_start_Eb = 7
SNR_dB_stop_Eb = 8
SNR_points = 1

SNR_dB_start_Es = SNR_dB_start_Eb + 10*np.log10(float(1000)/float(2004))
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10*np.log10(float(1000)/float(2004))

sigma_start = np.sqrt(1/(2*10**(float(SNR_dB_start_Es)/float(10))))
sigma_stop = np.sqrt(1/(2*10**(float(SNR_dB_stop_Es)/float(10))))

# Testing sigmas
#test_sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

test_sigmas = np.array([0.6310, 0.5623, 0.4994, 0.3985])
SNR_points = 4

SNRS = -10*np.log10(test_sigmas**2)

# =============================================================================
# Example showing the encoding and decoding of convolutional codes
# =============================================================================

# G(D) corresponding to the convolutional encoder
#generator_matrix = np.array([[03, 00, 02], [07, 04, 06]])
# Number of delay elements in the convolutional encoder
M = np.array([2])
generator_matrix = np.array([[05, 07]])
# Create trellis data structure
trellis = cc.Trellis(M, generator_matrix)

## RSC
M = np.array([3]) # Number of delay elements in the convolutional encoder
trellis = cc.Trellis(np.array([3]), np.array([[11,13]]))
print('trellis: cc.Trellis(np.array([3]), np.array([[11,13]]))')

M = np.array([1]) # Number of delay elements in the convolutional encoder
trellis = cc.Trellis(np.array([1]), np.array([[3,1]]),feedback=3)
print('trellis: cc.Trellis(np.array([1]), np.array([[3,1]]),feedback=3)')

#M = np.array([2])
#trellis = cc.Trellis(np.array([2]), np.array([[7,5]]),feedback=7)
#print('trellis: cc.Trellis(np.array([2]), np.array([[7,5]]),feedback=7)')

nb_errors = np.zeros(test_sigmas.shape)
map_nb_errors = np.zeros(test_sigmas.shape)
easy_ber = np.zeros(test_sigmas.shape)



# Traceback depth of the decoder
tb_depth = 10#5*(M.sum() + 1)

print('traceback depth: '+str(tb_depth))

for idx in xrange(SNR_points):
    print(idx)
    for iterations in xrange(iterations_number):  
#        print iterations
        message_bits = np.random.randint(0, 2, k)

        coded_bits = cc.conv_encode(message_bits, trellis)
#        print('coded_bits: ')
#        print(coded_bits)
        if NType == 'iid2':
            noise1 = test_sigmas[idx]*np.random.standard_normal(coded_bits.shape) # Define noise
            noise2 = test_sigmas[idx]*np.random.standard_normal(coded_bits.shape) # Define noise
            noise = (noise1+noise2)*0.5
            received = (2*coded_bits-1) + noise # Modulation plus noise
            
        if NType == 'iid':
            noise = test_sigmas[idx]*np.random.standard_normal(coded_bits.shape) # Define noise
            received = (2*coded_bits-1) + noise # Modulation plus noise
        elif NType == 'periodic':
            period = 2004
            
#            print 'period: ', period
            if period == 4 or period == 2004:            
                Tr_noise = test_sigmas[idx]*np.random.standard_normal([1,period])
                noise = np.repeat(Tr_noise,int(len(coded_bits)/period),axis=0).reshape(1,len(coded_bits))
#                print noise
            if period == 10:            
                noise = np.repeat(Tr_noise,int(len(coded_bits)/period),axis=0).reshape(1,2000)
                noise = np.concatenate((noise,Tr_noise[0][0:4].reshape(1,4)),axis=1)

            received = (2*coded_bits-1) + noise # Modulation plus noise
            received = received.reshape(len(coded_bits),)
        elif NType == 'arma':
            alpha = 0.5
            beta = np.sqrt(1-alpha**2)
            Tr_noise = np.zeros(coded_bits.shape)#test_sigmas[idx]*np.random.standard_normal([1,period])
            Tr_noise[0] = test_sigmas[idx]*np.random.standard_normal(1)

            for iii in xrange(1,len(coded_bits)):
                Tr_noise[iii]=alpha*Tr_noise[iii-1]+beta*test_sigmas[idx]*np.random.standard_normal(1)
            
            received = (2*coded_bits-1) + Tr_noise # Modulation plus noise
            received = received.reshape(len(coded_bits),)

#        print('received: ')
#        print(received)

        decoded_bits = cc.viterbi_decode(received.astype(float), trellis, tb_depth,decoding_type='unquantized')
# MAP
        received_odd = received[::2] # first, third, fifth, etc
        received_even = received[1::2] # second, fourth, sixth, etc 
        _, map_decoded_bits = commpy.channelcoding.map_decode(sys_symbols=received_odd.astype(float),  non_sys_symbols=received_even.astype(float), trellis = trellis, noise_variance = test_sigmas[idx]**2, L_int = np.zeros(received_odd.shape), mode = 'decode')

        myd = (received_odd>0)
        num_easy_ber = hamming_dist(message_bits[2:-int(M)], myd[2:-int(M)-int(M)])

        num_bit_errors = hamming_dist(message_bits[2:-int(M)], decoded_bits[2:-int(M)-int(M)])
        lenc = len(message_bits[2:-int(M)])
        map_num_bit_errors = hamming_dist(message_bits, map_decoded_bits[:-int(M)])

        nb_errors[idx]+= num_bit_errors
        easy_ber[idx]+= num_easy_ber
        print num_easy_ber
#        print num_bit_errors
        map_nb_errors[idx]+= map_num_bit_errors

    print('SNR: ' + str(SNRS[idx]))
    print('Viterbi BER: '+ str(nb_errors[idx]/float(lenc*iterations_number)))   
    print('MAP(BCJR) BER: '+ str(map_nb_errors[idx]/float(k*iterations_number))+'\n')
    print('Easy BER: '+ str(easy_ber[idx]/float(k*iterations_number))+'\n')


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
plt.savefig('Viterbi_and_MAP_'+str(NType)+'.png')

##
##file = open('depth_'+str(tb_depth)+'_Gaussian_iid_Viterbi_ber.txt',"w")
##file.write( nb_errors/float(k*iterations_number) )
##
##file2 = open('depth_'+str(tb_depth)+'_Gaussian_iid_Viterbi_snr.txt',"w")
##file2.write(str(SNRS))
##          
##file.close()
##file2.close()
