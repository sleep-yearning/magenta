import numpy as np
import math
import os
import pretty_midi as pm
from magenta.models.coconet.instrument_groups import groups


# selects instruments to use and converts single file into 4 note-arrays
def convert_file(pm_file, interpret_programs):
    p1, p2, p3, grouped = interpret_programs
    instruments = pm_file.instruments
    length = pm_file.get_end_time()
    # get total length of song times samplerate
    array_length = int(math.ceil(length * 100))

    rhythm_instruments = []        
    instruments1 = []
    instruments2 = []
    instruments3 = []

    if not grouped:
        for instrument in instruments:
            if not instrument.is_drum:
                if instrument.program == p1:
                    instruments1.append(instrument)
                elif instrument.program == p2:
                    instruments2.append(instrument)
                elif instrument.program == p3:
                    instruments3.append(instrument)
            else:
                rhythm_instruments.append(instrument)
    else:
        for group in groups:
            if p1 in group:
                group1 = group
            elif p2 in group:
                group2 = group
            elif p3 in group:
                group3 = group

        for instrument in instruments:
            if instrument.is_drum:
                rhythm_instruments.append(instrument)
            elif instrument.program in group1:
                instruments1.append(instrument)
            elif instrument.program in group2:
                instruments2.append(instrument)
            elif instrument.program in group3:
                instruments3.append(instrument)

    outcol = []
    # select the one with most messages
    if (instruments1):
        track1 = sorted(instruments1, key=lambda x: len(x.notes))[-1]
        col1 = instrument_to_column(track1, array_length, 1)
        outcol.append(col1)
    else:
        outcol.append(np.zeros(array_length))
    if (instruments2):
        track2 = sorted(instruments2, key=lambda x: len(x.notes))[-1]
        col2 = instrument_to_column(track2, array_length, 2)
        outcol.append(col2)
    else:
        outcol.append(np.zeros(array_length))
    if (instruments3):
        track3 = sorted(instruments3, key=lambda x: len(x.notes))[-1]
        col3 = instrument_to_column(track3, array_length, 3)
        outcol.append(col3)
    else:
        outcol.append(np.zeros(array_length))
    if (rhythm_instruments):
        rhythmtrack = sorted(rhythm_instruments, key=lambda x: len(x.notes))[-1]
        col4 = instrument_to_column(rhythmtrack, array_length, '(drums)')
        outcol.append(col4)
    else:
        outcol.append(np.zeros(array_length))
    return np.array(outcol).T


def instrument_to_column(instrument, array_length, song_row):
    if instrument:
        if song_row == '(drums)':
            print('Instrument: ', song_row, instrument.program)
        else:
            print('Instrument: ', song_row, pm.program_to_instrument_name(instrument.program))
        sampling_per_second = 100
        # using times gave artifacts, padding with zeros instead
        data = instrument.get_piano_roll(fs=sampling_per_second, times=None, include_drums=True)
        lowestloudest = np.zeros(array_length)
        for i in np.arange(data.shape[1]):
            row = data[:, i]
            # select the loudest (most important) values, if multiple, take lowest note (most of the time base note)
            lowestloudest[i] = min(np.argwhere(row == np.amax(row)))
    return lowestloudest


def main(path, grouped_instruments, p1, p2, p3):
    print('Main Instruments: ', pm.program_to_instrument_name(p1), pm.program_to_instrument_name(p2),
          pm.program_to_instrument_name(p3))
    converted_data = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.mid'):
            pm_file = pm.PrettyMIDI(os.path.join(path, filename), clip=True)
            print(filename)
            converted_data.append(convert_file(pm_file, [p1, p2, p3, grouped_instruments]))
    converted_array = np.asarray(converted_data)
    num_data_points=converted_array.shape[0]
    test_set_index=int(num_data_points/5)
    valid_set_index=int(test_set_index*4)
    np.savez(os.path.join(path, 'train_data.npz'), 
             test=converted_array[0:test_set_index], 
             train=converted_array[test_set_index:valid_set_index],
             valid=converted_array[valid_set_index:num_data_points])
    return converted_array

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=None, help='Path to Midi folder to be converted')
    parser.add_argument('--grouped', action='store_true', help='Groups instruments by type')
    parser.add_argument('program1', default=69, help='Interpret program for track 1, '
                                                     'as analyzed in analyze_instruments.py')
    parser.add_argument('program2', default=70, help='Interpret program for track 2, '
                                                     'as analyzed in analyze_instruments.py')
    parser.add_argument('program3', default=72, help='Interpret program for track 3, '
                                                     'as analyzed in analyze_instruments.py')
    args = parser.parse_args()
    main(args.path, args.grouped, args.program1, args.program2, args.program3)
