__author__ = 'yihanjiang'
from matplotlib import rc
import numpy as np

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.pylab as plt

# benchmarks
snrs =   [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
turbo_atn_v3_awgndec_ber  = [0.020477, 0.013664, 0.009777, 0.007183, 0.005807, 0.004106, 0.003304, 0.002552]

ntd_bl100_atn_v3_adapt_ber = [0.009775999002158642, 0.005022999830543995, 0.0026990005280822515, 0.0015460002468898892,
                              0.0008110000587999821, 0.0004420000792481005, 0.000245999894104898, 0.00013099999341648072]

turbofy_bl100_atn_v3_adapt_ber = [0.0033720002975314856, 0.00124520028475672, 0.00047400029143318534, 0.00019680021796375513,
                                 9.860002369852737e-05, 4.699998705997132e-05, 2.6599994842195883e-05, 1.3199996722803917e-05]






dta_q2_atn_enctrain2_dectrain_neg15_2_ber =   [0.002069319598376751, 0.0006864801398478448, 0.00026220001745969057, 0.00011157998233102262,
                                               5.56400373170618e-05, 2.784002026601229e-05, 1.2800018339476082e-05, 6.5200033532164525e-06,
                                               3.359998800078756e-06, 1.560000214340107e-06, 6.59999898289243e-07, 4.400000932491821e-07]

dta_q2_atn_enctrain2_dectrain_neg15_2_bler = [0.07207800000000002, 0.03429599999999998, 0.01626999999999999, 0.007919999999999986, 0.00415999999999999, 0.00212, 0.0009920000000000007, 0.0005260000000000003, 0.00027200000000000016, 0.00012200000000000007, 6.0000000000000035e-05, 3.6000000000000014e-05]


dta_cont_atn_enctrain2_dectrain_neg15_2_ber = [0.0018894205568358302, 0.0006292202742770314, 0.00022239997633732855, 9.039998985826969e-05, 4.220000118948519e-05, 1.8960005036205985e-05, 9.540002793073654e-06, 4.499998340179445e-06, 2.559999302320648e-06, 9.599998520570807e-07, 4.2000004896181053e-07, 2.200000039920269e-07]
dta_cont_atn_enctrain2_dectrain_neg15_2_bler = [0.06367799999999998, 0.028767999999999995, 0.01300399999999999, 0.006127999999999992, 0.0029859999999999934, 0.001410000000000001, 0.0007220000000000005, 0.0003260000000000002, 0.00019200000000000014, 6.800000000000004e-05, 3.200000000000001e-05, 1.6e-05]


# Mutual Informations
snrs_mi = [-1.0, 0.0, 1.0, 5.0, 20.0]

bi_awgn_mi = [0.416098173676, 0.48763823958, 0.554189773691,0.866028204142,1.00153286381 ]

neural_awgn_mi = [0.421056192663, 0.496989583177, 0.586175258253, 1.01267724435, 3.31720013783]


atn_snrs = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 5.0, 20.0]

bi_atn_mi = [0.130827592304, 0.325445025163,  0.496715010768,
    0.578405714594, 0.631198722814, 0.687616617396, 0.87211810785,  1.00042405286]

neural_atn_mi = [0.123288067184,0.331208399346, 0.536282350328,
                 0.629279122675,0.718414755903, 0.811364890355, 1.27528280211, 3.54548299723 ]


awgn_atn_mi  = [0.123989465406,  0.331626261371, 0.521727089853,
                0.626797531299, 0.708523707081, 0.807380726788,1.26878912213,  3.53345254615 ]


import matplotlib.pylab as plt

plt.figure(1)

plt.subplot(121)
plt.title('ATN Channel BER')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')

p0,  = plt.plot(snrs, turbo_atn_v3_awgndec_ber,'-.', label = r'{\sc Turbo}')
p1,  = plt.plot(snrs, turbofy_bl100_atn_v3_adapt_ber,'-o', label = r"{\sc DeepTurbo} Adaptivity")
p2,  = plt.plot(snrs, ntd_bl100_atn_v3_adapt_ber,'-x', label = r"{\sc NeuralBCJR} Adaptivity")

h1,  = plt.plot(snrs, dta_q2_atn_enctrain2_dectrain_neg15_2_ber[:8],'-', label = 'DTA-binary')
h2,  = plt.plot(snrs, dta_cont_atn_enctrain2_dectrain_neg15_2_ber[:8],'-+', label = 'DTA-continuous')

plt.legend(handles = [p0, p1, p2, h1,h2])
plt.grid()


plt.subplot(122)
plt.title('KSG Estimated MI')
plt.xlabel('SNR')
plt.ylabel('MI')

p10,  = plt.plot(snrs_mi, bi_awgn_mi, '-x',label = 'Binary-AWGN')
p11,  = plt.plot(snrs_mi, neural_awgn_mi, '-.',label = 'Continuous-AWGN')

p20,  = plt.plot(atn_snrs, bi_atn_mi, '-+',label = 'Binary-ATN')
p21,  = plt.plot(atn_snrs, neural_atn_mi, '-o', label = 'Continuous-ATN')
#p22,  = plt.plot(atn_snrs, awgn_atn_mi, '-+b', label = 'AWGN-ATN')


plt.legend(handles = [p10, p11,
                      p20, p21])
plt.grid()


plt.show()




