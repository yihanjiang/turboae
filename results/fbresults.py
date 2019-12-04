__author__ = 'yihanjiang'
import math
# Yihan's result on Turbo Code
# Turbo Code rate 1/3, block length 50.

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def snr_sigma2db(train_sigma):
    # print(train_sigma)
    return -20.0 * math.log(train_sigma, 10)


def convert_snr_to_ebno(this_input, coderate = 3.0):
    this_output = snr_sigma2db(snr_db2sigma(this_input)/math.sqrt(coderate))
    return this_output




turbo_SNR = [-2,-1,0,1,2, 3, 4]
turbo_Ebno = [convert_snr_to_ebno(item) for item in turbo_SNR]

turbo_ber = [9.87690000e-02,   3.89852000e-02,   8.17980000e-03, 8.04200000e-04,   2.56000000e-05, 2.64e-06, 4.8e-07]


# Turbo Code rate 1/3 block length 1000
t6_snr = [-1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0,
         2.5, 3.0, 3.5, 4.0]

t6_Ebno = [convert_snr_to_ebno(item) for item in t6_snr]

turbo757_bl1000_i6_ber = [0.02843181, 0.00209208, 0.00010128, 2.224e-05,
                       7.15e-06, 2.52e-06, 1.03e-06, 3.6e-07,
                       1.8e-07, 4.3e-08, 1.4e-08, 0.0]


# Hyeji's results: clean feedback channel (machine precision)
# DeepCode, rate 1/3, block length 50.
best_deepcode_SNR = [-2,-1, 0,1,2]
best_deepcode_Ebno = [convert_snr_to_ebno(item, coderate=3.0) for item in best_deepcode_SNR]
best_deepcode_ber = [ 0.0090905,0.0001296,1.99999999995e-06,9.99999999474e-08,3.99999999789e-08]



uncoded_rate2_snr =  [0.0, 2.0, 4.0, 6.0]
uncoded_rate2_Ebno = [convert_snr_to_ebno(item, coderate=2.0) for item in uncoded_rate2_snr ]
# soft message.
uncoded_rate2_ber =  [0.07837, 0.03751, 0.01258, 0.00256]
# hard message.

uncoded_rate2_ber = [0.0775, 0.0377, 0.0108, 0.0025]

conv_ble_rate2_snr =   [0.0, 2.0, 4.0, 6.0]
conv_ble_rate2_Ebno = [convert_snr_to_ebno(item, coderate=2.0) for item in conv_ble_rate2_snr ]
conv_ble_rate2_ber =  [0.106282, 0.014013, 0.0004104, 3.4e-06]


conv_ble_rate8_snr = [ -6.0, -4.0, -2.0, 0.0]
conv_ble_rate8_Ebno = [convert_snr_to_ebno(item, coderate=8.0) for item in conv_ble_rate8_snr ]
#conv_ble_rate8_ber =[ 0.11145, 0.0140, 0.000419, 3e-06]
conv_ble_rate8_ber =[0.21538, 0.07935, 0.01048, 0.0003]




print(conv_ble_rate2_Ebno)
print(conv_ble_rate8_Ebno)

import matplotlib.pylab as plt

shannon_limit_SNR = -11.70
shannon_limit_EbN0 = -1.59

# plt.figure(1)
# plt.title('DeepCode vs Turbo')
# plt.yscale('log')
# plt.xlabel('SNR')
# plt.ylabel('BER')
#
# b1 = plt.axvline(x=shannon_limit_SNR, label = 'Shannon Limit')
# h1,  = plt.plot(turbo_SNR, turbo_ber, 'b-',linewidth=3, label = 'Turbo Code: block length 50')
# #h2,  = plt.plot(t6_snr, turbo757_bl1000_i6_ber, 'g-',linewidth=2, label = 'Turbo Code: block length 1000')
#
# h3,  = plt.plot(best_deepcode_SNR, best_deepcode_ber,'k-x', linewidth=3, label = 'DeepCode: block length 50')
#
#
#
# plt.legend(handles = [b1, h1, h3])
# plt.grid()


plt.figure(2)
plt.title('DeepCode vs FEC, block length 50')
plt.yscale('log')
plt.xlabel('EbN0')
plt.ylabel('BER')

#b1 = plt.axvline(x=shannon_limit_EbN0, label = 'Shannon Limit')
h1,  = plt.plot(turbo_Ebno, turbo_ber, 'b-',linewidth=3, label = 'Turbo Code')
#h2,  = plt.plot(t6_Ebno, turbo757_bl1000_i6_ber, 'g-',linewidth=2, label = 'Turbo Code: block length 1000')

h3,  = plt.plot(best_deepcode_Ebno, best_deepcode_ber,'k-x', linewidth=3, label = 'DeepCode')

p1, = plt.plot(uncoded_rate2_Ebno, uncoded_rate2_ber,'g->', linewidth=3, label = 'Uncoded')

p2, = plt.plot(conv_ble_rate2_Ebno, conv_ble_rate2_ber,'r-<', linewidth=3, label = 'Conv Code: BT5 S=2')

p3, = plt.plot(conv_ble_rate8_Ebno, conv_ble_rate8_ber,'c-o', linewidth=3, label = 'Conv Code: BT5 S=8')

#plt.xlim([])


plt.legend(handles = [p1,p2,p3, h1, h3])
plt.grid()

plt.show()




