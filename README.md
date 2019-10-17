# Turbo AE
Turbo Autoencoder drafty code for paper: Y. Jiang, H. Kim, H. Asnani, S. Kannan, S. Oh, P. Viswanath, "Turbo Autoencoder: Deep learning based channel code for point-to-point communication channels" Conference on Neural Information Processing Systems (NeurIPS), Vancouver, December 2019

Note: this code is very drafty, and the final version will be out with camera ready paper. Paper camera-ready date is late October. Update ongoing.

Feel free to ask me any question! yij021@uw.edu

# What is new: 
(1) drafty paper (main.pdf and supplement.pdf) put in repo. Camera-ready paper under construction. Slides under construction.
(2) Pre-trained model under refining. Current *.pt in './models/' are not best model. (But you can finetune them easily) 

# Basic Usage:


# Run experiment:

(1) Test pre-trained model:
CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder dta_rate3_cnn -decoder dta_rate3_cnn  -enc_num_unit 100 -enc_num_layer 2  -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -group_norm_g 1 -train_channel_low 2.0 -train_channel_high 2.0 -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1  -train_dec_channel_low -1.5 -train_dec_channel_high 2.0  -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001  -num_block 100000 -batch_size 1000 -train_channel_mode group_norm -test_channel_mode group_norm -num_epoch 0 --print_test_traj -loss bce -init_nw_weight ./models/dta_cont_cnn2_cnn5_enctrain2_dectrainneg15_2.pt 

(2) Train from scratch:
Expect to run for one day on Nvidia 1080Ti.

CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder dta_rate3_cnn -decoder dta_rate3_cnn  -enc_num_unit 100 -enc_num_layer 2  -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -group_norm_g 1 -train_channel_low 2.0 -train_channel_high 2.0 -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1  -train_dec_channel_low -1.5 -train_dec_channel_high 2.0  -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001  -num_block 50000 -batch_size 500 -train_channel_mode group_norm -test_channel_mode group_norm -num_epoch 600 --print_test_traj -loss bce 

(3) Fine-tune on trained model:
CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder dta_rate3_cnn -decoder dta_rate3_cnn  -enc_num_unit 100 -enc_num_layer 2  -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -group_norm_g 1 -train_channel_low 2.0 -train_channel_high 2.0 -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1  -train_dec_channel_low -1.5 -train_dec_channel_high 2.0  -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001  -num_block 50000 -batch_size 500 -train_channel_mode group_norm -test_channel_mode group_norm -num_epoch 600 --print_test_traj -loss bce -init_nw_weight ./models/dta_cont_cnn2_cnn5_enctrain2_dectrainneg15_2.pt 





