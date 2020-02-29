import numpy as np
import math
import os
import pretty_midi as pm
from magenta.models.coconet.instrument_groups import rhythm_in_normal_channels, groups


# selects instruments to use and converts single file into 4 note-arrays
def convert_file(pm_file, interpret_programs):
    p1, p2, p3, grouped = interpret_programs
    instruments = pm_file.instruments
    length = pm_file.get_end_time()
    # get total length of song times samplerate
    array_length = int(math.ceil(length * 100))

    # cluster rhythmprograms
    rhythm_instruments = []
    for instrument in instruments:
        if instrument.is_drum:
            rhythm_instruments.append(instrument)
        elif (instrument.program in rhythm_in_normal_channels):
            rhythm_instruments.append(instrument)

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
    return outcol


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


def convert_folder(path, grouped_instruments):
    append_string = ''
    if grouped_instruments:
        append_string = 'grouped'
    try:
        p1, p2, p3, pR, rhythm_in_channel10 = np.load(path + 'programs' + append_string + '.npy')
    except IOError as err:
        print("IO error: {0}".format(err))
        return
    print('Main Instruments: ', pm.program_to_instrument_name(p1), pm.program_to_instrument_name(p2),
          pm.program_to_instrument_name(p3))
    converted_data = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.mid'):
            pm_file = pm.PrettyMIDI(path + filename, clip=True)
            print(filename)
            converted_data.append(convert_file(pm_file, [p1, p2, p3, rhythm_in_channel10]))
    converted_array = np.array(converted_data)
    num_data_points=converted_array.shape[0]
    test_set_index=int(num_data_points/5)
    valid_set_index=int(test_set_index*4)
    np.savez(path + 'train_data.npz', test=converted_array[0:test_set_index], train=converted_array[test_set_index:valid_set_index],
             valid=converted_array[valid_set_index:num_data_points])
    return converted_array


def main(arguments):
    convert_folder(arguments[0], arguments[1])


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(args)
