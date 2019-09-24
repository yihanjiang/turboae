__author__ = 'yihanjiang'

# benchmarks

turbo757_i6_bl100_snr = [-1.5, -1.0, -0.5, 0.0,
                         0.5, 1.0, 1.5, 2.0,
                         2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
turbo757_i6_bl100_ber = [0.056863, 0.02592, 0.010097, 0.002785,
                         0.000747, 0.00022, 6.9e-05, 2.832e-05,
                         1.1175e-05, 4.502e-06, 1.618e-06, 5.53e-07,
                         1.69e-07,5e-08 ]

turbo757_i6_bl100_bler = [0.526, 0.29300000000000004, 0.139, 0.05020000000000002,
                          0.018900000000000028, 0.007299999999999973, 0.0028000000000000247, 0.0010999999999999899,
                          0.0005629, 0.0002336, 8.48e-05, 2.8400000000039505e-05,
                          8.80000000003e-06, 2.59999999996e-06]

turbolte_i6_bl100_snr =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
turbolte_i6_bl100_ber = [0.06289789, 0.02797317, 0.00920123, 0.00218761,
                         0.00044006, 9.915e-05, 3.048e-05, 1.4e-05,
                         6.16e-06, 2.64e-06, 1.39e-06, 4.8e-07]
turbolte_i6_bl100_bler = [0.488499, 0.250826, 0.098333, 0.03151300000000001,
                          0.010387000000000035, 0.004279000000000033, 0.0020649999999999835, 0.0010949999999999571,
                          0.0005680000000000129, 0.000264000000000042, 0.00013700000000005375, 4.8000000000048004e-05]

# DeepTurbo as a benchmark

# deepturbo 757, iteration 6
turbofy_dec_757_snr_tf5 = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
turbofy_dec_757_ber_tf5 =  [0.06876672804355621, 0.034134820103645325, 0.013278993777930737, 0.0039285761304199696,
                            0.0009306119754910469, 0.00019915630400646478, 4.617767262971029e-05, 1.2138983947806992e-05,
                            3.7910908758931328e-06, 1.2122184216423193e-06, 3.544449214132328e-07, 1.2888885692063923e-07 ]

turbofy_dec_757_bler_tf5 = [0.6815720000000027, 0.44594700000000115, 0.22712100000000046, 0.08711500000000033,
                            0.02625300000000005, 0.006943999999999906, 0.0018749999999999615, 0.0005277000000000013,
                            0.00017189999999999323, 5.730000000000004e-05, 1.7000000000000013e-05, 6.300000000000004e-06]
# deepturbo LTE, iteration 6
turbofylte_i6_ft5_snr =   [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,2.5, 3.0, 3.5, 4.0]
turbofylte_i6_ft5_ber =  [0.08252130448818207, 0.04216403886675835, 0.016300421208143234, 0.004570276476442814,
                          0.0009356688242405653, 0.00015390923363156617, 2.5644580091466196e-05, 4.8999922910297755e-06,
                          1.4333338640426518e-06, 3.6666673963736685e-07, 1.7888896763906814e-07, 4.666667763331134e-08]

turbofylte_i6_ft5_bler = [0.7063299999999955, 0.4673839999999981, 0.23551400000000208, 0.08631199999999992,
                          0.023135000000000402, 0.004996000000000043, 0.0010729999999999833, 0.00024299999999999927,
                           7.700000000000006e-05, 2.800000000000001e-05, 1.2500000000000009e-05, 3.7000000000000014e-06]

###########################################################################################################################
# DTA results
###########################################################################################################################





snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# RNN only results
dta_level2_rnn_enctrain2_dectrainneg15_2_ber = [0.08769059926271439, 0.04307781159877777, 0.016747096553444862, 0.005380249582231045, 0.001673299353569746, 0.0005798998172394931, 0.0002428000298095867, 0.00010475004091858864, 4.6200017095543444e-05, 1.770000199030619e-05, 8.499999239575118e-06, 3.5499992918630596e-06]

# CNN Our results


dta_cont_cnnenc_grudec_enctrain2_dectrainneg15_2_ber = [0.10225459188222885, 0.048412203788757324, 0.018568996340036392, 0.00535960029810667, 0.0013811998069286346, 0.00043360001291148365, 0.00012820000119972974, 4.7399993491126224e-05, 1.4400003237824421e-05, 3.400000196052133e-06, 9.999999974752427e-07, 3.9999997625272954e-07]


dta_q2_cnnenc_grudec_enctrain2_dectrainneg15_2_ber =[0.07956141233444214, 0.03532780334353447, 0.012283999472856522, 0.0038170008920133114,
                                                     0.0012933999532833695, 0.0004459999909158796, 0.00016799998411443084, 6.600000051548705e-05,
                                                     2.5599989385227673e-05, 7.599997843499295e-06, 1.9999999949504854e-06, 7.999999525054591e-07]


dta_q2_cnnenc_cnndec_enctrain2_dectrainneg15_2_ber = [0.17315320670604706, 0.10201200097799301, 0.04709819331765175, 0.01594560034573078,
                                                      0.004049600102007389, 0.0007532000890932977, 0.0001596000511199236, 4.0999995690071955e-05, 1.700000029813964e-05, 5.599999894911889e-06, 2.1999999262334313e-06, 3.9999997625272954e-07]
dta_q2_cnnenc_cnndec_enctrain2_dectrainneg15_2_ber = [0.11324860155582428, 0.056740593165159225, 0.02226019836962223, 0.006556401029229164,
                                                      0.0015670000575482845, 0.0003893998800776899, 0.00012340000830590725, 4.0599985115695745e-05,
                                                      1.359999987471383e-05, 5.199999345175456e-06, 1.1999999287581886e-06, 5.999999643790943e-07]

###############################################################
# CNN 2 layer encoder, CNN 5 layer decoder, DTA.
###############################################################

dta_cont_cnn2enc_cnn5dec_enctrain1_dectrainneg15_2_ber =  [0.09983271360397339, 0.04594481363892555, 0.015701422467827797, 0.0040611871518194675,
                                                           0.0008357250480912626, 0.00019907507521566004, 6.008748459862545e-05, 1.871250969998073e-05,
                                                           5.912505457672523e-06, 1.7874991726785083e-06, 5.624999062092684e-07, 3.7499997773693394e-08]
dta_q2_cnn2enc_cnn5dec_enctrain1_dectrainneg15_2_ber =  [0.6655637500000001, 0.4240062500000007, 0.21046000000000023, 0.08428624999999997, 0.030414999999999987, 0.010892499999999986, 0.003978749999999979, 0.0013050000000000004, 0.0004200000000000003, 0.0001325000000000001, 4.125000000000003e-05, 3.75e-06]


dta_cont_cnn2enc_cnn5dec_enctrain2_dectrainneg15_2_ber = [0.10074236243963242, 0.04645926132798195, 0.016037607565522194, 0.004073908552527428,
                                                          0.0008630399242974818, 0.00019923502986785024, 6.059997758711688e-05, 1.9379973309696652e-05,
                                                          6.014996415615315e-06, 1.7300009176324238e-06, 4.1999996369668224e-07, 9.00000216574881e-08]

dta_q2_cnn2enc_cnn5dec_enctrain2_dectrainneg15_2_ber = [0.08792993426322937, 0.03961125388741493, 0.01354303676635027, 0.003595138667151332,
                                                        0.0008670896058902144, 0.0002506198943592608, 8.951997006079182e-05, 2.9970024115755223e-05,
                                                        1.1290004295005929e-05, 3.6000019463244826e-06, 1.070000166691898e-06, 3.000000106112566e-07]

dta_cont_cnn2enc_cnn5dec_enctrain3_dectrainneg15_2_ber = [0.09881149232387543, 0.04532505199313164, 0.015476478263735771, 0.003931159619241953,
                                                          0.000831500394269824, 0.0001962550013558939, 5.925497316638939e-05, 2.0169980416540056e-05,
                                                          6.294998001976637e-06, 1.7650010022407514e-06, 4.300000568946416e-07, 9.000001455206075e-08]


dta_cont_cnn2enc_cnn5dec_enctrain0_dectrain0_ber =[0.08134345710277557, 0.03552451357245445, 0.011664831079542637, 0.0030416203662753105,
                                                   0.0007227200549095869, 0.00021446991013363004, 7.218997780000791e-05, 2.530002166167833e-05,
                                                   8.7200114649022e-06, 2.4999987999763107e-06, 7.700000423938036e-07, 2.1000003869175998e-07]

dta_q2_cnn2enc_cnn5dec_enctrain0_dectrain0_ber =[0.08684908598661423, 0.03950152173638344, 0.01391039788722992, 0.0038691104855388403,
                                                  0.0010155497584491968, 0.0003109500976279378, 0.00011194997932761908, 4.18375275330618e-05,
                                                  1.5125009667826816e-05, 5.262500962999184e-06, 1.56249996052793e-06, 6.12499832186586e-07]

dta_cont_cnn2enc_cnn5dec_enctrainneg05_dectrainneg05_ber =  [0.05700160935521126, 0.026030899956822395, 0.010004100389778614, 0.0037539000622928143,
                                                            0.0014785000821575522, 0.000622799969278276, 0.00024505003239028156, 9.615000453777611e-05,
                                                            3.580000702640973e-05, 1.4799999007664155e-05, 5.249999048828613e-06, 2.4999997094710125e-06]



dta_cont_cnn2enc_cnn5dec_enctrainneg15_dectrainneg15_ber  =  [0.047043804079294205, 0.025237491354346275, 0.01260764803737402, 0.006137849763035774,
                                                              0.002895999699831009, 0.0013556497870013118, 0.0006062999600544572, 0.0002708999963942915,
                                                              0.00011360000644344836, 4.029999763588421e-05, 1.7149997802334838e-05, 5.949999831500463e-06]

dta_q2_cnn2enc_cnn5dec_enctrainneg15_dectrainneg15_ber  =  [0.08075735718011856, 0.03741989657282829, 0.013716748915612698, 0.004008599556982517,
                                                            0.0011138498084619641, 0.00034955001319758594, 0.0001240499987034127, 4.6399996790569276e-05,
                                                            1.689999407972209e-05, 6.099999609432416e-06, 2.1999994714860804e-06, 6.999999868639861e-07]

# DTA results:
dta_cont_cnn2enc_cnn5dec_combined_ber = [0.047043804079294205, 0.025237491354346275,  0.010004100389778614, 0.0030416203662753105,
                                        0.0007227200549095869,0.00019167498976457864, 5.911249900236726e-05,  1.819999568699859e-05,
                                        5.587504801951582e-06, 1.7300009176324238e-06, 4.1999996369668224e-07, 9.00000216574881e-08]

dta_q2_cnn2enc_cnn5dec_combined_ber = [0.08075735718011856, 0.03741989657282829, 0.01354303676635027, 0.0035666588228195906,
                                        0.0008611400262452662, 0.0002465699799358845, 8.57499981066212e-05, 2.9970024115755223e-05,
                                       1.0350004231440835e-05, 3.2000019463244826e-06, 1.070000166691898e-06, 3.000000106112566e-07]

# with more complicated CNN encoder
dta_cont_cnn5_cnn5_enctrain2_dectrainneg15_2_ber =[0.11056305468082428, 0.05583880841732025, 0.021506845951080322, 0.005767849739640951,
                                                  0.001228200038895011, 0.00022149999858811498, 5.360001159715466e-05, 1.6099997083074413e-05,
                                                4.699999863078119e-06, 1.2000000424450263e-06, 4.999999987376214e-07, 0.0]

dta_cont_cnn5_cnn8_enctrain2_dectrainneg15_2_ber = [0.1292179971933365, 0.06680618971586227, 0.026148391887545586, 0.007060298230499029,
     0.0015614001313224435, 0.0002323999797226861, 4.2000003304565325e-05, 1.3200001376389991e-05,
     3.3000001167238224e-06, 7.999999525054591e-07, 2.9999998218954715e-07, 1.9999998812636477e-07]



dta_cont_cnn5_cnn10_enctrain3_dectrainneg15_2_ber =[0.14237380027770996, 0.07572893053293228, 0.03047935850918293, 0.008855819702148438,
                                                    0.0018523395992815495, 0.00029650007490999997, 4.383999839774333e-05, 1.0660014595487155e-05,
                                                    2.8799993287975667e-06, 8.199999683711212e-07, 7.999999951380232e-08, 0.0]

dta_cont_cnn5_cnn10_enctrain1_dectrainneg15_2_ber = [0.11016751825809479, 0.05326661095023155, 0.01864680089056492, 0.004769998602569103,
                                                     0.0008188999490812421, 0.00016039996990002692, 3.829998604487628e-05, 1.4199996257957537e-05,
                                                     2.8000004022032954e-06, 1.7000000980260666e-06, 4.999999987376214e-07, 0.0]


dta_cont_cnn_combined_ber = [0.047043804079294205, 0.025237491354346275,  0.010004100389778614, 0.0030416203662753105,
                            0.0007227200549095869,0.00016039996990002692, 3.829998604487628e-05, 1.0660014595487155e-05,
                            2.8799993287975667e-06, 7.999999525054591e-07, 7.999999951380232e-08, 0]


dta_q2_cnn5_cnn10_enctrain2_dectrainneg15_2_ber = [0.0999421551823616, 0.04713989049196243, 0.01698349602520466, 0.0046648988500237465,
                                                   0.0011045997962355614, 0.00026489997981116176, 8.860000525601208e-05, 3.0499999411404133e-05,
                                                   9.700001101009548e-06, 2.699999868127634e-06, 1.1999999287581886e-06, 2.9999998218954715e-07]

dta_q2_cnn5_cnn10_enctrain3_dectrainneg15_2_ber = [0.10980260372161865, 0.053133293986320496, 0.019489100202918053, 0.005129499826580286,
     0.0011463999981060624, 0.00023979999241419137, 6.090001988923177e-05, 2.1099989680806175e-05,
     5.6999988373718224e-06, 1.6000000186977559e-06, 5.999999643790943e-07, 1.9999998812636477e-07]


dta_q2_cnn2enc_cnn5dec_combined_ber = [0.08075735718011856, 0.03741989657282829, 0.01354303676635027, 0.0035666588228195906,
                                        0.0008611400262452662, 0.00023979999241419137, 6.090001988923177e-05, 2.1099989680806175e-05,
                                    5.6999988373718224e-06, 1.6000000186977559e-06, 5.999999643790943e-07, 1.9999998812636477e-07]




# BLERs

dta_cont_cnn2enc_cnn5dec_enctrain2_dectrainneg15_2_bler =[0.6667735000000005, 0.4258410000000012, 0.21239300000000005, 0.08461900000000021,
                                                          0.03029899999999994, 0.010856499999999944, 0.003988999999999977, 0.0013614999999999838,
                                                          0.00043950000000000033, 0.0001255000000000001, 3.2500000000000024e-05, 8.000000000000003e-06]

dta_q2_cnn2enc_cnn5dec_enctrain2_dectrainneg15_2_bler = [0.6662359999999999, 0.4307619999999997, 0.22268199999999988, 0.096248,
                                                         0.03869800000000008, 0.015362999999999991, 0.006246999999999979, 0.0023629999999999875,
                                                         0.0008620000000000007, 0.0002710000000000002, 0.00010500000000000009, 2.900000000000002e-05]

dta_cont_cnn2enc_cnn5dec_enctrain3_dectrainneg15_2_bler = [0.6603455000000009, 0.41917950000000004, 0.20791449999999975, 0.08336900000000028, 0.02970049999999999, 0.010917999999999949, 0.003947499999999985, 0.001407499999999983, 0.00044300000000000036, 0.0001225000000000001, 3.100000000000002e-05, 7.500000000000004e-06]



dta_cont_cnn2enc_cnn5dec_enctrain0_dectrain0_bler =  [0.641712999999999, 0.40324099999999996, 0.2007309999999997, 0.08383600000000012,
                                                      0.032335000000000017, 0.012571999999999995, 0.0048169999999999775, 0.0017979999999999888,
                                                      0.0006210000000000005, 0.00018900000000000015, 5.600000000000004e-05, 1.7000000000000007e-05]


dta_q2_cnn2enc_cnn5dec_enctrain0_dectrain0_bler =  [0.6834962500000001, 0.45406999999999975, 0.2449762500000003, 0.11114249999999988,
                                                      0.046681250000000014, 0.019503749999999983, 0.007919999999999988, 0.003123749999999985,
                                                      0.001185000000000001, 0.0004162500000000003, 0.00012500000000000008, 5.125000000000004e-05]



dta_cont_cnn2enc_cnn5dec_enctrainneg15_dectrainneg15_bler  =  [0.755155, 0.5886149999999999, 0.40657999999999994, 0.25509999999999994,
                                                               0.14547499999999994, 0.078055, 0.038000000000000006, 0.018129999999999993,
                                                               0.008054999999999996, 0.0030300000000000023, 0.001360000000000001, 0.0004950000000000003]

dta_q2_cnn2enc_cnn5dec_enctrainneg15_dectrainneg15_bler  =  [0.6655000000000001, 0.44377500000000003, 0.24486999999999995, 0.11482999999999999,
                                                             0.04960000000000002, 0.021174999999999996, 0.008639999999999995, 0.0034450000000000023,
                                                             0.001345000000000001, 0.0005050000000000003, 0.0001800000000000001, 4.5e-05]


# combined result
dta_cont_cnn2enc_cnn5dec_combined_bler =[0.641712999999999, 0.40324099999999996, 0.2007309999999997, 0.08383600000000012,
                                         0.03029899999999994, 0.010844999999999992, 0.003974999999999994, 0.0013614999999999838,
                                        0.00043500000000000033, 0.0001255000000000001, 3.2500000000000024e-05, 8.000000000000003e-06]


dta_q2_cnn5enc_cnn5dec_combined_bler =  [0.7030399999999999, 0.46197999999999995, 0.23621000000000003, 0.09296000000000003,
                                         0.03197999999999999, 0.011180000000000002, 0.0036700000000000027, 0.0014600000000000008,
                                         0.00042000000000000023, 0.00012000000000000004, 1e-05, 2e-05]


dta_q2_cnn5_cnn10_enctrain3_dectrainneg15_2_bler = [0.6662359999999999, 0.4307619999999997, 0.22268199999999988, 0.096248,
                                                0.03869800000000008, 0.015362999999999991, 0.006246999999999979, 0.0023629999999999875,
                                                         0.0008620000000000007, 0.0002710000000000002, 0.00010500000000000009, 2.900000000000002e-05]


dta_cont_cnn5_cnn10_enctrain1_dectrainneg15_2_bler = [0.6799100000000005, 0.43408999999999986, 0.21200999999999992, 0.07948000000000005,
                                                      0.023769999999999986, 0.007460000000000006, 0.0026300000000000017, 0.0009400000000000007,
                                                      0.00021000000000000012, 0.00010000000000000002, 4e-05, 0.0]
dta_cont_cnn5_cnn10_enctrain3_dectrainneg15_2_bler = [0.7322760000000006, 0.495104, 0.2550440000000001, 0.09700200000000005,
                                                      0.028527999999999967, 0.007475999999999956, 0.002149999999999999, 0.0006660000000000005,
                                                      0.00020400000000000016, 5.4000000000000025e-05, 6e-06, 0.0]

dta_q2_cnn5_cnn10_enctrain3_dectrainneg15_2_bler = [0.69434, 0.46179000000000014, 0.23988000000000004, 0.10313000000000004,
                                                    0.03963999999999998, 0.01542999999999999, 0.005960000000000004, 0.0021600000000000018, 0.0007900000000000006, 0.00022000000000000006, 0.00010000000000000002, 3e-05]





dta_cont_cnn_combined_bler = [0.641712999999999, 0.40324099999999996, 0.2007309999999997, 0.07948000000000005,
                                         0.023769999999999986,  0.007475999999999956, 0.002149999999999999, 0.0006660000000000005,
                                        0.00020400000000000016, 5.4000000000000025e-05, 2e-05, 8.000000000000003e-06]




import matplotlib.pylab as plt

plt.figure(1)
plt.subplot(121)
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')

p1, = plt.plot(turbo757_i6_bl100_snr[:12], turbo757_i6_bl100_ber[:12],'-x', label = 'Turbo-757')
p2, = plt.plot(turbolte_i6_bl100_snr, turbolte_i6_bl100_ber,'-+', label = 'Turbo-LTE')

h3, = plt.plot(snrs, dta_q2_cnn2enc_cnn5dec_combined_ber, '-.', label = 'TurboAE-binary')

h4, = plt.plot(snrs, dta_cont_cnn_combined_ber, '-o', label = 'TurboAE-continuous')


plt.legend(handles =[p1,p2,
                      h3, h4])
plt.grid()

plt.subplot(122)
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BLER')

p1, = plt.plot(turbo757_i6_bl100_snr[:12], turbo757_i6_bl100_bler[:12],'-x', label = 'Turbo-757')
p2, = plt.plot(turbolte_i6_bl100_snr, turbolte_i6_bl100_bler,'-+', label = 'Turbo-LTE')

h3, = plt.plot(snrs, dta_q2_cnn5enc_cnn5dec_combined_bler, '-.', label = 'TurboAE-binary')

h4, = plt.plot(snrs, dta_cont_cnn_combined_bler, '-o', label = 'TurboAE-continuous')


plt.legend(handles =[p1,p2,
                      h3, h4])
plt.grid()


# put DeepTurbo-CNN into the picture.

turbofy757_cnn_l5_k5_c100_i6_ft5_snr =  [-1.5, -1.0, -0.5, 0.0,
                                         0.5, 1.0, 1.5, 2.0,
                                         2.5, 3.0, 3.5, 4.0]
turbofy757_cnn_l5_k5_c100_i6_ft5_ber =  [0.07986578345298767, 0.04472022131085396, 0.019834663718938828, 0.0069457776844501495,
                                         0.0019557776395231485, 0.00047288884525187314, 0.00011200005246791989, 2.6999947294825688e-05,
                                         8.200009688152932e-06, 2.4444443624815904e-06, 4.666667052788398e-07, 2.2222224060897133e-07]

turbofy757_cnn_l5_k5_c100_i6_ft5_bler =  [0.8021399999999997, 0.6015199999999999, 0.36574000000000007, 0.16998000000000002,
                                          0.06197999999999995, 0.019340000000000013, 0.0052400000000000025, 0.0013840000000000011,
                                          0.0004160000000000003, 0.00013400000000000009, 2.800000000000001e-05, 1.2e-05]

plt.figure(2)
plt.subplot(121)
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')

p1, = plt.plot(turbo757_i6_bl100_snr[:12], turbo757_i6_bl100_ber[:12],'-x', label = 'Turbo-757')
p2, = plt.plot(turbolte_i6_bl100_snr, turbolte_i6_bl100_ber,'-+', label = 'Turbo-LTE')

s1, = plt.plot(turbofy757_cnn_l5_k5_c100_i6_ft5_snr[:12], turbofy757_cnn_l5_k5_c100_i6_ft5_ber[:12],'-x', label = 'Turbo-757-CNN')

h3, = plt.plot(snrs, dta_q2_cnn2enc_cnn5dec_combined_ber, '-.', label = 'DTA-binary CNN-enc CNN-dec')

h4, = plt.plot(snrs, dta_cont_cnn2enc_cnn5dec_combined_ber, '-o', label = 'DTA-continuous CNN-enc CNN-dec')


plt.legend(handles =[p1, p2, s1,
                      h3, h4])
# plt.grid()
#
# plt.subplot(122)
# plt.yscale('log')
# plt.xlabel('SNR')
# plt.ylabel('BLER')
#
# p1, = plt.plot(turbo757_i6_bl100_snr[:12], turbo757_i6_bl100_bler[:12],'-x', label = 'Turbo-757')
# p2, = plt.plot(turbolte_i6_bl100_snr, turbolte_i6_bl100_bler,'-+', label = 'Turbo-LTE')
#
# s1, = plt.plot(turbofy757_cnn_l5_k5_c100_i6_ft5_snr[:12], turbofy757_cnn_l5_k5_c100_i6_ft5_bler[:12],'-x', label = 'DeepTurbo-757-CNN')
#
#
# h3, = plt.plot(snrs, dta_q2_cnn2enc_cnn5dec_combined_bler, '-.', label = 'DTA-binary CNN-enc CNN-dec')
#
# h4, = plt.plot(snrs, dta_cont_cnn2enc_cnn5dec_combined_bler, '-o', label = 'DTA-continuous CNN-enc CNN-dec')
#
#
# plt.legend(handles =[p1, p2,s1,
#                       h3, h4])
# plt.grid()


#wienna 5G simulator

wienna_uncoded_ber = [0.200053000000008	,0.186427058823540,	0.172560470588247,	0.158606235294129,	0.144778352941188,	0.130900823529421,	0.117213705882360,	0.103913764705885	,0.0911395882352947,	0.0788604117647032,	0.0673152941176380,	0.0565631764705695]


CodingSchemes = {'Uncoded', 'LDPC', 'Turbo',  'Polar','TBC'};


wienna_LDPC_ber =[0.185257000000001,	0.126287600000000,	0.0656968000000000,	0.0244306000000000,
                  0.00632200000000000,	0.00113110000000000,	0.000145500000000000,	1.60600000000000e-05,
                  1.98000000000000e-06	,1.13000000000000e-07	,2.00000000000000e-09,	0]


wienna_turbo_Linear_LogMAP_ber    =[0.138865681818182,	0.0874613636363636,	0.0410465909090909,	0.0134468181818182,
                                    0.00284227272727273,	0.000456590909090909,	4.67386363636364e-05,	3.87500000000000e-06,
                                    4.31818181818182e-07,	3.75000000000000e-07,	0,	0]

wienna_Polar_ber    =[0.273416000000001	,0.220505400000001,	0.178270400000000,	0.150902400000000,
                      0.135288800000000,	0.127563800000000,	0.123696400000000,	0.122156800000000,	0.121017400000000,	0.120826600000000,	0.120262800000000,	0.120048800000000]
wienna_conv_ber    =	[0.210119999999999, 0.141556399999999	,0.0832093999999996	,0.0417980000000000	,
                         0.0181558000000000	,
                        0.00653098999999965,	0.00209175000000000	,0.000604900000000001,	0.000149760000000000,
                        3.57500000000000e-05,	7.04000000000000e-06,	1.09000000000000e-06]

wienna_Turbo_ber  = [0.1743  ,  0.1362  ,  0.08767,    0.0417 ,
                     0.0134 ,   0.0025069 ,   0.0002891   , 0.0000289, 0.0000018, 0.0000003, 0, 0]

wienna_Turbo_ber_i6 = [ 0.17507 ,   0.1387 ,   0.08942,    0.0425 ,
                        0.01438 ,   0.00352 ,   0.00047,   4.649e-5,
                        3.49e-6, 8e-7]


wienna_Polar_ber = [0.3447,    0.2454 ,   0.1512 ,   0.0586 ,
                    0.0157  , 0.0027483, 2.8090e-04 , 2.4800e-05,
                    4.0000e-06, 3.7500e-07, 0,0  ]


rate2_uncoded = [0.158798500000006,	0.130827400000001,	0.103956099999995,	0.0787929999999961,	0.0565983999999973,
                 0.0376115999999978,	0.0229921999999990,	0.0126032000000002,	0.00597410000000019]
rate2_ldpc    = [0.236799700000000,	0.173161900000002,	0.0692287000000000,	0.00860540000000000,	0.000280200000000000,
                 3.27000000000000e-06,	0,	0,	0]
rate2_tbcc    = [0.367910700000000,	0.240660500000000,	0.0945826999999989,	0.0174622000000000,	0.00163520000000000,
                 7.96000000000000e-05,	2.70000000000000e-06,	4.00000000000000e-08,	0]
rate2_turbo_   = [0.161327272727273,	0.103844318181818,	0.0317329545454546	,0.00311136363636364,
                  9.65909090909091e-05,	1.35227272727273e-06	,0	,0,	0]

# new results:
rate2_polar   = [0.4164 ,   0.2556,    0.064489,   0.0090320, 0.0001292,        2.5e-6   ,      0,         0    ,     0 ]



# trainenc on 4dB
dta_rate2_continuous_ber = [0.2279827892780304, 0.10073654353618622, 0.02051619626581669, 0.0015372189227491617,
                            8.094005170278251e-05, 7.340023330471013e-06, 1.2399999604895129e-06,
                            1.000000082740371e-07, 9.99999993922529e-09]
dta_rate2_continuous_bler =  [0.9332559999999995, 0.6468209999999989, 0.22118200000000013, 0.034192000000000014, 0.004399999999999962, 0.0005430000000000004, 8.400000000000006e-05, 6e-06, 1e-06]

# trainenc on 0-4dB
dta_rate2_continuous_ber =  [0.16251453757286072, 0.05767969414591789, 0.009777599014341831, 0.001118999789468944,
                             0.00017774998559616506, 3.279999873484485e-05, 5.499999133462552e-06,
                             1.2999997807128238e-06, 1.4999999109477358e-07]
dta_rate2_continuous_bler =  [0.9167500000000004, 0.6182700000000001, 0.23535000000000006, 0.05892000000000003, 0.013159999999999998, 0.0025750000000000018, 0.00045500000000000027, 8.5e-05, 1e-05]

dta_rate2_continuous_ber = [0.16251453757286072, 0.05767969414591789, 0.009777599014341831, 0.001118999789468944,
                            8.094005170278251e-05, 7.340023330471013e-06, 1.2399999604895129e-06,
                            1.000000082740371e-07, 9.99999993922529e-09]


dta_rate2_binary_ber = [0.1971702128648758, 0.08434964716434479, 0.01813255064189434, 0.002120749792084098, 0.00028764994931407273, 5.73499892198015e-05, 1.2600000445672777e-05, 5.149998742126627e-06, 1.500000053056283e-06]
dta_rate2_binary_bler =  [0.9456299999999996, 0.7019400000000003, 0.304435, 0.07979000000000003, 0.018664999999999987, 0.0044950000000000025, 0.0009900000000000006, 0.00034500000000000015, 9.000000000000003e-05]

plt.figure(3)
plt.subplot(121)
plt.title('Code rate 1/3')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')

p1, = plt.plot(turbo757_i6_bl100_snr[:10], turbo757_i6_bl100_ber[:10],'-x', label = 'Turbo')
#p2, = plt.plot(turbolte_i6_bl100_snr, turbolte_i6_bl100_ber,'-+', label = 'Turbo-LTE')

p2, = plt.plot(snrs[:10], dta_q2_cnn2enc_cnn5dec_combined_ber[:10], '-.', label = 'TurboAE-binary')

p3, = plt.plot(snrs[:10], dta_cont_cnn_combined_ber[:10], '-o', label = 'TurboAE-continuous')

#s1, = plt.plot(turbo757_i6_bl100_snr[:12], wienna_uncoded_ber[:12],'-x', label = 'Wienna5G: uncoded')
p4, = plt.plot(turbo757_i6_bl100_snr[:10], wienna_conv_ber[:10],'-+', label = 'TBCC')
#s3, = plt.plot(turbo757_i6_bl100_snr[:12], wienna_turbo_Linear_LogMAP_ber[:12],'-x', label = 'Wienna5G: Turbo 6 iteration, Linear LogMAP (bl=88, not 100!)')
p5, = plt.plot(turbo757_i6_bl100_snr[:10], wienna_LDPC_ber[:10],'-<', label = 'LDPC')
p6, = plt.plot(turbo757_i6_bl100_snr[:10], wienna_Polar_ber[:10],'-x', label = 'Polar')

plt.legend(handles =[p2, p3, p4, p5, p1, p6])
plt.grid()

snrs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
plt.subplot(122)
plt.title('Code rate 1/2')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')

p1, = plt.plot(snrs[:6], rate2_turbo_[:6],'-x', label = 'Turbo')
p2, = plt.plot(snrs[:6], dta_rate2_binary_ber[:6],'-.', label = 'TurboAE-binary')
p3, = plt.plot(snrs[:6], dta_rate2_continuous_ber[:6],'-o', label = 'TurboAE-continuous')
p4, = plt.plot(snrs[:6], rate2_tbcc[:6],'-+', label = 'TBCC')
p5, = plt.plot(snrs[:6], rate2_ldpc[:6],'-<', label = 'LDPC')
p6, = plt.plot(snrs[:6], rate2_polar[:6],'-x', label = 'Polar')




plt.legend(handles =[ p2,
                      p3, p4, p5,p6,  p1])
plt.grid()


dta_cont_no_inter_ber =  [0.15685176849365234, 0.11168290674686432, 0.07274758815765381, 0.04336030036211014, 0.022550400346517563, 0.01062140055000782, 0.004485601559281349, 0.0016831997781991959, 0.0005844999686814845, 0.00018069999350700527, 5.190000229049474e-05, 1.5799994798726402e-05]
dta_cont_no_inter_bler = [0.9673899999999999, 0.9045100000000005, 0.7824700000000006, 0.60095, 0.3937200000000001, 0.22206000000000006, 0.10654, 0.04432999999999999, 0.01674999999999999, 0.005880000000000004, 0.0016900000000000012, 0.0005800000000000003]


cnn_ae_ber =  [0.07059461623430252, 0.04456420615315437, 0.026049993932247162, 0.013904601335525513,
               0.006724798120558262, 0.0028732004575431347, 0.001142599736340344, 0.00038819992914795876,
               0.00013740001304540783, 4.179999086773023e-05, 1.599999814061448e-05, 3.000000106112566e-06]



plt.figure(4)
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')
snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

p1, = plt.plot(snrs[:10], dta_q2_cnn2enc_cnn5dec_combined_ber[:10], '-.', label = 'TurboAE-binary')
p2, = plt.plot(snrs[:10], dta_cont_cnn_combined_ber[:10], '-o', label = 'TurboAE-continuous')

p31, = plt.plot(turbo757_i6_bl100_snr[:10], turbo757_i6_bl100_ber[:10],'-x', label = 'Turbo-757-6i')
p32, = plt.plot(turbo757_i6_bl100_snr[:10], wienna_Turbo_ber_i6[:10],'-x', label = 'Turbo-Wienna5G-6i')


p4, = plt.plot(turbo757_i6_bl100_snr[:10], wienna_conv_ber[:10],'-+', label = 'TBCC')
p5, = plt.plot(turbo757_i6_bl100_snr[:10], wienna_LDPC_ber[:10],'-<', label = 'LDPC')
p6, = plt.plot(snrs[:10], wienna_Polar_ber[:10],'-x', label = 'Polar')

p7, = plt.plot(turbo757_i6_bl100_snr[:10], cnn_ae_ber[:10],'--', label = 'CNN-AE')
#p7, = plt.plot(turbo757_i6_bl100_snr[:10], dta_cont_no_inter_ber[:10],'->', label = 'TurboAE-no interleaver')


plt.legend(handles =[p1, p2, p31,p32, p4, p5, p6, p7])
plt.grid()


plt.show()





