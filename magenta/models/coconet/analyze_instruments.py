import numpy as np
import os
import pretty_midi as pm
from magenta.models.coconet.instrument_groups import groups, rhythm_in_normal_channels


# finds the most common (groups of) instruments for a given folder of midi files
# and writes them into a file in that folder
def find_frequent_programs(folder, grouped):
    print(grouped)
    filename_specifier = ''
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
        filename_specifier = 'grouped'
        for group in groups:
            groupset = dict((k, programs[k]) for k in group if k in programs)
            max_key = max(groupset, key=lambda key: groupset[key])

            for key in group:
                if key is not max_key:
                    programs[max_key] += programs.pop(key, 0)

    # search for most frequent rhythm instrument
    max_rhythm_program = 0
    rhythm_program = None
    rhythm_in_channel10 = True
    # in usual drum channel instruments
    if rhythm:
        sorted_rhythm = {k: v for k, v in sorted(rhythm.items(), key=lambda item: item[1], reverse=True)}
        rhythm_program = next(iter(sorted_rhythm))
        max_rhythm_program = sorted_rhythm[rhythm_program]
    # in normal channel instruments which are typical rhythm instruments
    for special_program in rhythm_in_normal_channels:
        if special_program in programs and programs[special_program] > max_rhythm_program:
            max_rhythm_program = programs[special_program]
            rhythm_program = special_program
            rhythm_in_channel10 = False
    # remove instrument from normal selection if it was selected as rhythm program
    if rhythm_program and not rhythm_in_channel10:
        programs.pop(rhythm_program, None)
    #select 3 most frequent programs
    sorted_programs = iter({k: v for k, v in sorted(programs.items(), key=lambda item: item[1], reverse=True)})
    p1 = next(sorted_programs)
    p2 = next(sorted_programs)
    p3 = next(sorted_programs)

    print(folder)
    ret_array = [p1, p2, p3, rhythm_program, rhythm_in_channel10]
    print(ret_array)

    np.save(folder + 'programs' + filename_specifier + '.npy', ret_array)
    return ret_array


def main(path, grouped):
    find_frequent_programs(path, grouped)

#if called with an extra argument, grouped analysis will be done
if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) < 2:
        main(args[0], False)
    elif args[1]:
        main(args[0], True)
