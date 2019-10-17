# Documents for replicate results in paper.
Note that this is noise-driven learning procedure, sometimes it take time, parameter tuning, and luck to reach specific performance. 
Try run with different random seed, the performance differs!


# Replicate Figure 1 Right.

## Benchmarks:
LDPC, Polar and Turbo are from MATLAB simulation, is from Vienna Simulator, which requires licence. The detailed setup is shown in paper appendix.

The Turbo Code benchmark is from Commpy, as in folder ./commpy/turbo_code_benchmarks.py. For block length 100, use the following command:
    python turbo_codes_benchmark.py -enc1 7 -enc2 5 -feedback 7 -M 2 -num_block 100000 -block_len 100 -num_cpu 4 -snr_test_start -1.5 -snr_test_end 2.0 -snr_points 8

By default it runs Turbo-757, but if you need Turbo-LTE, simply run:
    python turbo_codes_benchmark.py -enc1 13 -enc2 11 -feedback 13 -M 3 -num_block 100000 -block_len 100 -num_cpu 4 -snr_test_start -1.5 -snr_test_end 2.0 -snr_points 8

Note that when running higher SNR, it is suggested to use more `-num_block` and more `-num_cpu` to get better estimation.


## Training TurboAE from scratch:

### Get SNR=1dB good continuous code: TurboAE-continuous.
Start from scratch by running: 
    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 500 --print_test_traj -loss bce

This will take 1-1.5 days on nVidia 1080 Ti to train. After converged, it will show the model is saved in `./tmp/*.pt`.
Then incrase batch size by a factor of 2, and train for another 100 epochs:
    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 100 --print_test_traj -loss bce -init_nw_weight ./tmp/*.pt

When saturates, increase the batch size, till you hit some memory limit. Then reduce the learning rate for both encoder and decoder.


### Fine-tune at other SNRs
Change encoder training SNR to testing SNR, for example, change to `-train_enc_channel_low 2.0 -train_enc_channel_high 2.0 ` will train good code at 2dB. 
I find the training SNR for decoder fixed to `-1.5 to 2dB` seems works best.

### Binarize Code via STE, to generate TurboAE-binary.
Use a well-trained continuous code as initialization, and finetune with STE as:
    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm_ste -test_channel_mode block_norm_ste -num_epoch 200 --print_test_traj -loss bce -init_nw_weight ./tmp/*.pt

Also use increasing batch size, and reducing learning rate till converges. Train model for each SNR. 
Typically after 200-400 epochs, STE-trained model converges. 