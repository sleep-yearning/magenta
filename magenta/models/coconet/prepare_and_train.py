from magenta.models.coconet import analyze_instruments, midi_folder_transversion, coconet_train
import tensorflow as tf
import numpy as np

def prepare(path,grouped):        
    interpret_instruments = analyze_instruments.find_frequent_programs(path, grouped)
    converted_data = midi_folder_transversion.convert_folder(path, grouped)
    
    min_pitch = min(np.amin(file) for file in converted_data)
    max_pitch = max(np.amax(file) for file in converted_data)
    return interpret_instruments,min_pitch,max_pitch

def train(path,epochs,modelpath,interpret_instruments,min_pitch,max_pitch):
    
    train_args = [
    'logdir=' + path + 'log/',
    'log_process=True',
    'data_dir=' + path,
    # Save instruments with model config
    'program1=' + str(interpret_instruments[0]),
    'program2=' + str(interpret_instruments[1]),
    'program3=' + str(interpret_instruments[2]),
    'program4=' + str(interpret_instruments[3]),
    'rhythmProgramChannel10=' + str(interpret_instruments[4]),
    'min_pitch='+str(min_pitch),
    'max_pitch='+str(max_pitch),
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
    'interleave_split_every_n_layers=2']

    tf.app.run(main=coconet_train.main(), argv=train_args)

def main(path,epochs,grouped,modelpath):
    instruments, min_pitch, max_pitch = prepare(path,grouped)
    train(path,epochs,modelpath,instruments,min_pitch,max_pitch)

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) < 4:
        raise ValueError('not enough arguments, user must specify in order: '+
                         'Midi Folder path, epochs to train, are instruments grouped, model save path')
    else:
        main(args)
