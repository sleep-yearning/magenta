path = '/home/noah/Document/Studiumkram/ACM/Projekt/coconet/abba/'
epochs = 0
grouped = True
modelpath = '/home/noah/Document/Studiumkram/ACM/Projekt/coconet/models/abba/'

from magenta.models.coconet import analyze_instruments, midi_folder_transversion, coconet_train
import tensorflow as tf

interpret_instruments = analyze_instruments.find_frequent_programs(path, grouped)
converted_data = midi_folder_transversion.convert_folder(path)
min_pitch = min(converted_data)
max_pitch = max(converted_data)

train_args = [
    'logdir=' + path + 'log/',
    'log_process=True',
    'data_dir=' + path,
    # Save instruments with model config
    'program1=' + interpret_instruments[0],
    'program2=' + interpret_instruments[1],
    'program3=' + interpret_instruments[2],
    'program4=' + interpret_instruments[3],
    'rhythmProgramChannel10=' + interpret_instruments[4],
    'min_pitch='+min_pitch,
    'max_pitch='+max_pitch,
    # Data preprocessing.
    'dataset=TestData',
    'crop_piece_len=32',
    'separate_instruments=True',
    'quantization_level=0.125',
    # Hyperparameters.
    'maskout_method=orderless',
    'num_layers=32',
    'num_filters=64',
    'use_residual',
    'batch_size=10',
    'use_sep_conv=True',
    'architecture=dilated',
    'num_dilation_blocks=1',
    'dilate_time_only=False',
    'repeat_last_dilation_level=False',
    'num_pointwise_splits=2',
    'interleave_split_every_n_layers=2'
]


def main():
    tf.app.run(main=coconet_train.main(), argv=train_args)


if __name__ == '__main__':
    main()
