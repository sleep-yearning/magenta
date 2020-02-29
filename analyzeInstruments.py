import numpy as np
import os
import pretty_midi as pm
from InstrumentGroups import groups, rhythm_in_normal_channels


# finds the most common (groups of) instruments for a given folder of midi files
# and writes them into a file in that folder
def findFrequentPrograms(folder, grouped):
    print(grouped)
    filenameSpecifier = ''
    programs = {}
    rhythm = {}

    for filename in os.listdir(folder):
        if filename.endswith('.mid'):
            mid = pm.PrettyMIDI(folder + filename, clip=True)
            for instrument in mid.instruments:
                # get drumkit definitions separately
                if instrument.is_drum:
                    if instrument.program in rhythm:
                        rhythm[instrument.program] += 1
                    else:
                        rhythm[instrument.program] = 1
                # count up instruments occurences
                else:
                    if instrument.program in programs:
                        programs[instrument.program] += 1
                    else:
                        programs[instrument.program] = 1
    if grouped:
        filenameSpecifier = 'grouped'
        for group in groups:
            groupset = dict((k, programs[k]) for k in group if k in programs)
            maxKey = max(groupset, key=lambda key: groupset[key])

            for key in group:
                if key is not maxKey:
                    programs[maxKey] += programs.pop(key, 0)

    # search for most frequent rhythm instrument
    maxRhythmProgram = 0
    rhythmProgram = None
    rhythmInChannel10 = True
    # in usual drum channel instruments
    if rhythm:
        sortedRhythm = {k: v for k, v in sorted(rhythm.items(), key=lambda item: item[1], reverse=True)}
        rhythmProgram = next(iter(sortedRhythm))
        maxRhythmProgram = sortedRhythm[rhythmProgram]
    # in normal channel instruments which are typical rhythm instruments
    for specialProgram in rhythm_in_normal_channels:
        if specialProgram in programs and programs[specialProgram] > maxRhythmProgram:
            maxRhythmProgram = programs[specialProgram]
            rhythmProgram = specialProgram
            rhythmInChannel10 = False
    # remove instrument from normal selection if it was selected as rhythm program
    if rhythmProgram and not rhythmInChannel10:
        programs.pop(rhythmProgram, None)
    #select 3 most frequent programs
    sortedPrograms = iter({k: v for k, v in sorted(programs.items(), key=lambda item: item[1], reverse=True)})
    p1 = next(sortedPrograms)
    p2 = next(sortedPrograms)
    p3 = next(sortedPrograms)

    print(folder)
    retArray = [p1, p2, p3, rhythmProgram, rhythmInChannel10]
    print(retArray)

    np.save(folder + 'programs' + filenameSpecifier + '.npy', retArray)
    return retArray


def main(path, grouped):
    findFrequentPrograms(path, grouped)

#if called with an extra argument, grouped analysis will be done
if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) < 2:
        main(args[0], False)
    elif args[1]:
        main(args[0], True)
