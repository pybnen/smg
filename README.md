# smg

## Requirements

pytorch  
fluidsynth  
midi2audio  
pypianoroll  
sacred  

## Run experiments

[Sacred](https://github.com/IDSIA/sacred) is used to manage experiments.

### LSTM network

> `python -m smg.experiments.train_lstm`

Use `sacred` commands to print or change configuration ([Sacred Docs: Command-Line Interface](https://sacred.readthedocs.io/en/stable/command_line.html)).
Or change the `configs/config.json` to overwrite configs, which might be more convenient then using the commandline.

#### Configs

**Data Loader**
`batch_size`: Size of batches returned by the data loader
`n_workers`: Number of loader worker processes

**Dataset**
`in_seq_length` Sequence length of input for LSTM
`step_size` Step size used to generate sequences from one sample
`out_seq_length` Output length of prediction

The dataset can be loaded from a pregenerated obj file or by providing a directory with the sample files.
Each option has different configurations.

*From directory*:
`data_dir`  Directory containing sample files
`valid_split` Ratio of how many sequences should be used as validation set (possible values from 0 to 1)

`instruments` List of instruments that should be used (possible values 'drums', 'piano', 'guitar', 'bass', 'strings')
`lowest_pitch` Lowest pitch to be used
`n_pitches` The size of the pitch range, starting from the lowest
`beat_resolution` The resolution of the samples

*From pregenerated obj file*:
`data_obj_file` Path to data obj file
`valid_step_size` Step size for validation dataset

**Model**
`hidden_size` Hidden size of LSTM layer
`num_layers` Number of stacked LSTMs
`dense_layer_hiddens` Additional hidden layers for the feedforward network, a list containing the hidden size for each hidden layer 

**Training process**
`num_epochs` Number of epochs to train
`lr` Learning rate of optimizer
`checkpoint` Path to checkpoint file to continue training