import numpy as np
import math
import os
import pretty_midi as pm
from InstrumentGroups import rhythm_in_normal_channels, groups


# selects instruments to use and converts single file into 4 note-arrays
def convert_file(pm_file, interpretPrograms):
    p1, p2, p3, grouped = interpretPrograms
    instruments = pm_file.instruments
    length = pm_file.get_end_time()
    # get total length of song times samplerate
    arrayLength = int(math.ceil(length * 100))

    # cluster rhythmprograms
    rhythmInstruments = []
    for instrument in instruments:
        if instrument.is_drum:
            rhythmInstruments.append(instrument)
        elif (instrument.program in rhythm_in_normal_channels):
            rhythmInstruments.append(instrument)

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
                rhythmInstruments.append(instrument)
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
        col1 = instrumentToColumn(track1, arrayLength, 1)
        outcol.append(col1)
    else:
        outcol.append(np.zeros(arrayLength))
    if (instruments2):
        track2 = sorted(instruments2, key=lambda x: len(x.notes))[-1]
        col2 = instrumentToColumn(track2, arrayLength, 2)
        outcol.append(col2)
    else:
        outcol.append(np.zeros(arrayLength))
    if (instruments3):
        track3 = sorted(instruments3, key=lambda x: len(x.notes))[-1]
        col3 = instrumentToColumn(track3, arrayLength, 3)
        outcol.append(col3)
    else:
        outcol.append(np.zeros(arrayLength))
    if (rhythmInstruments):
        rhythmtrack = sorted(rhythmInstruments, key=lambda x: len(x.notes))[-1]
        col4 = instrumentToColumn(rhythmtrack, arrayLength, '(drums)')
        outcol.append(col4)
    else:
        outcol.append(np.zeros(arrayLength))
    return outcol


def instrumentToColumn(instrument, arrayLength, songRow):
    if instrument:
        if songRow == '(drums)':
            print('Instrument: ', songRow, instrument.program)
        else:
            print('Instrument: ', songRow, pm.program_to_instrument_name(instrument.program))
        sampling_per_second = 100
        # using times gave artifacts, padding with zeros instead
        data = instrument.get_piano_roll(fs=sampling_per_second, times=None, include_drums=True)
        lowestloudest = np.zeros(arrayLength)
        for i in np.arange(data.shape[1]):
            row = data[:, i]
            # select the loudest (most important) values, if multiple, take lowest note (most of the time base note)
            lowestloudest[i] = min(np.argwhere(row == np.amax(row)))
    return lowestloudest


def convertFolder(path, groupedInstruments):
    appendString = ''
    if groupedInstruments:
        appendString = 'grouped'
    try:
        p1, p2, p3, pR, rhythmInChannel10 = np.load(path + 'programs' + appendString + '.npy')
    except IOError as err:
        print("IO error: {0}".format(err))
        return
    print('Main Instruments: ', pm.program_to_instrument_name(p1), pm.program_to_instrument_name(p2),
          pm.program_to_instrument_name(p3))
    convertedData = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.mid'):
            pmFile = pm.PrettyMIDI(path + filename, clip=True)
            print(filename)
            convertedData.append(convert_file(pmFile, [p1, p2, p3, rhythmInChannel10]))
    convertedArray = np.array(convertedData)
    dataPoints=convertedArray.shape[0]
    testSet=dataPoints/5
    validSet=testSet*4
    np.savez(path + 'trainData.npz', test=convertedArray[0:testSet], train=convertedArray[testSet:validSet],
             valid=convertedArray[validSet:dataPoints])
    return convertedArray


def main(arguments):
    convertFolder(arguments[0], arguments[1])


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(args)
