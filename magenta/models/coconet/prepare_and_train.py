from magenta.models.coconet import analyze_instruments, midi_folder_transversion, coconet_train
import tensorflow as tf
import numpy as np
import os


def prepare(path, grouped):
    interpret_instruments = analyze_instruments.main(path, grouped)
    converted_data = midi_folder_transversion.main(path, grouped, interpret_instruments[0], interpret_instruments[1],
                                                   interpret_instruments[2])

    interpret_instruments.append(int(min(np.amin(file) for file in converted_data)))
    interpret_instruments.append(int(max(np.amax(file) for file in converted_data)))

    filename = 'programs.npy'
    if grouped:
        filename = 'programs_grouped.npy'

    np.save(os.path.join(path, filename), np.asarray(interpret_instruments))
    return interpret_instruments


def train(path, epochs=0, grouped=True, model_name="New", num_layers=32, num_filters=64, use_residual=True, batch_size=10,
          use_sep_conv=True, architecture='dilated', num_dilation_blocks=1, dilate_time_only=False,
          repeat_last_dilation_level=False, num_pointwise_splits=2, interleave_split_every_n_layers=2):
    # TODO: test and fine tune, maybe use more hparams
    train_args = {
        'dataset': 'TrainData',
        # Data preprocessing.
        'crop_piece_len': 32,
        # Hyperparameters.
        'num_epochs': epochs,
        'num_layers': num_layers,
        'num_filters': num_filters,
        'use_residual': use_residual,
        'batch_size': batch_size,
        'use_sep_conv': use_sep_conv,
        'architecture': architecture,
        'checkpoint_name': None,
        'num_dilation_blocks': num_dilation_blocks,
        'dilate_time_only': dilate_time_only,
        'repeat_last_dilation_level': repeat_last_dilation_level,
        'num_pointwise_splits': num_pointwise_splits,
        'interleave_split_every_n_layers': interleave_split_every_n_layers
    }

    logdir = os.path.join(path, model_name + '_checkpoint')
    log_progress = True

    coconet_train.main(train_args, path, grouped, logdir, log_progress)


def main(path, epochs, grouped, model_name):
    prepare(path, grouped)
    train(path, epochs, grouped, model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=None, help='Path to folder where midis are found and train_data.npz + '
                                                   'programs(_grouped).npy will be stored.')
    parser.add_argument('--epochs', default=0, help='Optionally set epochs to train, defaults to no limit.')
    parser.add_argument('--grouped', action='store_true', help='If passed, instruments will be grouped by type.')
    parser.add_argument('--model_name', default='model', help='Optionally set the name of the trained model.')
    args = parser.parse_args()
    main(args.path, args.epochs, args.grouped, args.model_name)
