path = '/home/noah/Document/Studiumkram/ACM/Projekt/coconet/abba/'
epochs = 0
grouped = True
modelpath = '/home/noah/Document/Studiumkram/ACM/Projekt/coconet/models/abba/'

import analyzeInstruments
import midiFolderTransversion
from magenta.models.coconet import coconet_train
import tensorflow as tf

interpretInstruments = analyzeInstruments.findFrequentPrograms(path, grouped)
convertedData = midiFolderTransversion.convertFolder(path)
minPitch = min(convertedData)
maxPitch = max(convertedData)

train_args = [
    'logdir=' + path + 'log/',
    'log_process=True',
    'data_dir=' + path,
    # Save instruments with model config
    'program1=' + interpretInstruments[0],
    'program2=' + interpretInstruments[1],
    'program3=' + interpretInstruments[2],
    'program4=' + interpretInstruments[3],
    'rhythmProgramChannel10=' + interpretInstruments[4],
    'min_pitch='+minPitch,
    'max_pitch='+maxPitch,
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
