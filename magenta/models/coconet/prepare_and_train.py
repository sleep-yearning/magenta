from magenta.models.coconet import analyze_instruments, midi_folder_transversion, coconet_train
import tensorflow as tf
import numpy as np
import os

def prepare(path,grouped):        
    interpret_instruments = analyze_instruments.main(path, grouped)
    converted_data = midi_folder_transversion.main(path, grouped, interpret_instruments[0], interpret_instruments[1],
                                                   interpret_instruments[2])
    
    interpret_instruments.append(int(min(np.amin(file) for file in converted_data)))
    interpret_instruments.append(int(max(np.amax(file) for file in converted_data)))

    filename='programs.npy'
    if grouped:
        filename='programs_grouped.npy'

    np.save(os.path.join(path, filename), np.asarray(interpret_instruments))
    return interpret_instruments

def train(path,epochs,modelpath,grouped):

    # TODO: test and fine tune, maybe use more hparams
    train_args = {
    'dataset' : 'TrainData',
    # Data preprocessing.
    'crop_piece_len' : 32,
    # Hyperparameters.
    'num_epochs' : epochs,
    'num_layers' : 32,
    'num_filters' : 64,
    'use_residual' : True,
    'batch_size' : 10,
    'use_sep_conv' : True,
    'architecture' : 'dilated',
    'checkpoint_name' : None,
    'num_dilation_blocks' : 1,
    'dilate_time_only' : False,
    'repeat_last_dilation_level' : False,
    'num_pointwise_splits' : 2,
    'interleave_split_every_n_layers' : 2
    }

    logdir= os.path.join(path, 'logs/')
    log_progress=True

    coconet_train.main(train_args,path,grouped,logdir,log_progress)

def main(path, modelpath, epochs, grouped):
    prepare(path,grouped)
    train(path, modelpath, epochs, grouped)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=None, help='Path to folder where midis are found and train_data.npz + '
                                                   'programs(_grouped).npy will be stored.')
    parser.add_argument('--model_path', default=None, help='Optional path to store model somewhere else than in '
                                                           'input path')
    parser.add_argument('--epochs', default=0, help='Optionally set epochs to train, defaults to no limit.')
    parser.add_argument('--grouped', action='store_true', help='If passed, instruments will be grouped by type.')
    args = parser.parse_args()
    main(args.path, args.model_path, args.epochs, args.grouped)
