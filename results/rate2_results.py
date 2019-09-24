__author__ = 'yihanjiang'

# rate 2 matlab
# legend('Uncoded', 'Convolutional (MAX-Log-MAP)','Turbo (Linear-Log-MAP, 8 iter)','LDPC (PWL-Min-Sum, 16 iter)','Polar (CRC-List-SC, 8 list)', 'location', 'northeast');
# CodingSchemes = {'Uncoded', 'LDPC', 'Turbo',  'Polar','TBC'};


snrs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
rate2_uncoded = [0.158798500000006,	0.130827400000001,	0.103956099999995,	0.0787929999999961,	0.0565983999999973,
                 0.0376115999999978,	0.0229921999999990,	0.0126032000000002,	0.00597410000000019]
rate2_ldpc    = [0.236799700000000,	0.173161900000002,	0.0692287000000000,	0.00860540000000000,	0.000280200000000000,
                 3.27000000000000e-06,	0,	0,	0]
rate2_tbcc    = [0.367910700000000,	0.240660500000000,	0.0945826999999989,	0.0174622000000000,	0.00163520000000000,
                 7.96000000000000e-05,	2.70000000000000e-06,	4.00000000000000e-08,	0]
rate2_turbo_   = [0.161327272727273,	0.103844318181818,	0.0317329545454546	,0.00311136363636364,
                  9.65909090909091e-05,	1.35227272727273e-06	,0	,0,	0]


dta_rate2_continuous_ber = [0.2279827892780304, 0.10073654353618622, 0.02051619626581669, 0.0015372189227491617, 8.094005170278251e-05, 7.340023330471013e-06, 1.2399999604895129e-06, 1.000000082740371e-07, 9.99999993922529e-09]
dta_rate2_continuous_bler =  [0.9332559999999995, 0.6468209999999989, 0.22118200000000013, 0.034192000000000014, 0.004399999999999962, 0.0005430000000000004, 8.400000000000006e-05, 6e-06, 1e-06]




import matplotlib.pylab as plt

plt.figure(1)
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')

p1, = plt.plot(snrs, rate2_uncoded,'-x', label = 'Uncoded')
p2, = plt.plot(snrs, rate2_ldpc,'-x', label = 'LDPC (rate=1/2)')
p3, = plt.plot(snrs, rate2_tbcc,'-x', label = 'TBCC (rate=1/2)')
p4, = plt.plot(snrs, rate2_turbo_,'-x', label = 'Turbo (rate=0.44)')

p5, = plt.plot(snrs, dta_rate2_continuous_ber,'-+', label = 'DTA-continuous')
plt.legend(handles =[p1, p2,
                      p3, p4, p5])
plt.grid()
plt.show()

