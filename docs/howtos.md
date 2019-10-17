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

    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 800 --print_test_traj -loss bce

This will take 1-1.5 days on nVidia 1080 Ti to train. Take some rest, hiking or watch a season of Anime during waiting.
After converged, it will show the model is saved in `./tmp/*.pt`.
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
Typically after 200-400 epochs, STE-trained model converges. Note: this actually replicates **Figure 4 right**.
 
 
# Replicate Figure 4 left
Run 5 times below commands for RNN:
    
    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_rnn -decoder TurboAE_rate3_rnn -enc_num_unit 100 -enc_num_layer 2 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 800 --print_test_traj -loss bce

Run 5 times below commands for CNN:

    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 800 --print_test_traj -loss bce

You can take the printout log (saved in `./logs/*.txt`), to help you plot the graph.

# Replicate Figure 5 left: Coding Gain.
Turbo performance is from the benchmark using Turbo-757. 
CNN-AE is done through another implementation, which will be public soon. 
TurboAE is same as above, just changed the `-block_len`. 
Note that TurboAE with large block length `-block_len 1000` is hard to train, as it takes a long of memory. 
It will be appreciated to share some well-trained model for longer block length, if you have more resource.

# Replicate Figure 5 right: Interleaver.

To run with the same training interleaving, simple do `-is_interleave 1`, to use:

    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 0 --print_test_traj -loss bce -is_interleave 1 -init_nw_weight ./models/dta_cont_cnn2_cnn5_enctrain2_dectrainneg15_2.pt

To run with a random interleaver, use `-is_interleave n`, as `n` is the random seed. To not use interleaver, use `-is_interleave 0` argument.


# Replicate Figure 6
Start from AWGN converged TurboAE-continuous, and use `-channel atn` and `-channel ge_awgn`. Fine-tune till converge.
The ATN result (Figure 6 left) actually I didn't train a lot, so the performance could even be better at ATN channel.
Also note as `ge_awgn` is a sequential dependent channel, simulation is much slower than iid channels.

