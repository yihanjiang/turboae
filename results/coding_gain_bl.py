__author__ = 'yihanjiang'

'''
CNN-AE
bl1000:
Test SNR 2.0 with ber  0.0002462019619997591 with bler 0.1166499999999998
bl200:
Test SNR 2.0 with ber  0.0002650997485034168 with bler 0.02597999999999999
bl100 :
Test SNR 2.0 with ber  0.00031172001035884023 with bler 0.01574000000000001
bl50:
Test SNR 2.0 with ber  0.00038431986467912793 with bler 0.009369999999999922
bl20
Test SNR 2.0 with ber  0.0007009999244473875 with bler 0.006760000000000005


TurboAE:
bl1000:
7.499999810534064e-06
bl200:
Test SNR 2.0 with ber  9.349998435936868e-06 with bler 0.001440000000000001
bl100
Test SNR 2.0 with ber  1.6999996660160832e-05 with bler 0.0011000000000000007
bl50:
Test SNR 2.0 with ber  8.740001067053527e-05 with bler 0.0017150000000000012
bl20
Test SNR 2.0 with ber  0.0023757496383041143 with bler 0.008619999999999994

CUDA_VISIBLE_DEVICES=2 python3.6 main.py -encoder dta_rate3_cnn -decoder dta_rate3_cnn  -enc_num_unit 100 -dec_num_unit 100 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1  -code_rate_k 1 -code_rate_n 3 -group_norm_g 1 -train_channel_low 2.0 -train_channel_high 2.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1  -train_dec_channel_low -1.5 -train_dec_channel_high 2.0  -group_norm_g 1 -is_same_interleaver 1 -num_train_enc 1  -num_train_dec 5 -dec_lr 0.00001 -enc_lr 0.00001  -num_block 200000 -batch_size 2000  -enc_num_layer 3 -dec_num_layer 5  -num_epoch 10 --print_test_traj -init_nw_weight ./tmp/torch_model_456044.pt -init_nw_weight ./tmp/torch_model_744301.pt -num_epoch 0 -snr_test_start 2.0 -snr_test_end 2.5 -snr_points 2  -block_len 1000 -num_block 20000

'''
bl = [20, 50, 100, 200, 1000]


cnn_ae_2dB_ber = [0.0007009999244473875,0.00038431986467912793 ,0.00031172001035884023,0.0002650997485034168,  0.0002462019619997591]

turbo_ae_2dB_ber = [0.0023757496383041143, 8.740001067053527e-05, 1.6999996660160832e-05,  9.349998435936868e-06,  6.24999984211172e-06]


turbo_2dB_ber = [ 0.0007485,  9.4e-05, 3.6e-05, 2.35e-06,  3.9e-07 ]



import matplotlib.pylab as plt


plt.figure(1)

plt.title('Coding Gain of different block length')
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Block Length')
plt.ylabel('BER at 2dB')


p1,  = plt.plot(bl, cnn_ae_2dB_ber,'-x', label = 'CNN-AE')
p2,  = plt.plot(bl, turbo_ae_2dB_ber,'->', label = 'TurboAE')
p3,  = plt.plot(bl, turbo_2dB_ber,'-<', label = 'Turbo')

plt.legend(handles = [p1, p2, p3])

plt.grid()

plt.show()




















