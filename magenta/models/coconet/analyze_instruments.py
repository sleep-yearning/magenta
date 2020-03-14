import numpy as np
import os
import pretty_midi as pm
from magenta.models.coconet.instrument_groups import groups

# finds the most common (groups of) instruments for a given 
# folder of midi files
def main(folder, grouped):
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
    # for grouped analysis: sum groups score into 
    # its strongest contributer
    if grouped:
        for group in groups:
            groupset = dict((k, programs[k]) for k in group if k in programs)
            # most frequent instrument in group
            if (groupset):
                max_key = max(groupset, key=lambda key: groupset[key])

                for key in group:
                    if key is not max_key:
                        programs[max_key] += programs.pop(key, 0)

    # select most frequent rhythm instrument
    rhythm_program = None
    if rhythm:
        sorted_rhythm = {k: v for k, v in sorted(
                rhythm.items(), key=lambda item: item[1], reverse=True)}
        rhythm_program = next(iter(sorted_rhythm))
    
    # select 3 most frequent programs
    sorted_programs = iter({k: v for k, v in sorted(
                programs.items(), key=lambda item: item[1], reverse=True)})
    p1 = next(sorted_programs)
    p2 = next(sorted_programs)
    p3 = next(sorted_programs)

    print(folder)
    ret_instruments = [p1, p2, p3, rhythm_program]
    print(ret_instruments)
    return ret_instruments

#if called with the 'grouped' argument, grouped instrument analysis will be done
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=None, help='File path to'
                                            ' folder that is to be analyzed.')
    parser.add_argument('--grouped', action='store_true', 
                        help='Should instruments be grouped by class')
    args=parser.parse_args()
    main(args.path,args.grouped)
