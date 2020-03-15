# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to evaluate a dataset fold under a model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from magenta.models.coconet import lib_data
from magenta.models.coconet import lib_evaluation
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_util
import numpy as np
import tensorflow.compat.v1 as tf

EVAL_SUBDIR = 'eval_stats'


def main(ckpt, evaldir, unit, chronological, ensemble_size, sample_path,
         folder, index, data_dir):
    checkpoint_dir = ckpt
    if not checkpoint_dir:
        # If a checkpoint directory is not specified, see if there is only one
        # subdir in this folder and use that.
        possible_checkpoint_dirs = tf.gfile.ListDirectory(evaldir)
        possible_checkpoint_dirs = [
            i for i in possible_checkpoint_dirs if
            tf.gfile.IsDirectory(os.path.join(evaldir, i))]
        if EVAL_SUBDIR in possible_checkpoint_dirs:
            possible_checkpoint_dirs.remove(EVAL_SUBDIR)
        if len(possible_checkpoint_dirs) == 1:
            checkpoint_dir = os.path.join(
                evaldir, possible_checkpoint_dirs[0])
            tf.logging.info('Using checkpoint dir: %s', checkpoint_dir)
        else:
            raise ValueError(
                'Need to provide a path to checkpoint directory or use an '
                'eval_logdir with only 1 checkpoint subdirectory.')
    wmodel = lib_graph.load_checkpoint(checkpoint_dir)
    if evaldir is None:
        raise ValueError(
            'Set flag eval_logdir to specify a path for saving eval statistics.')
    else:
        eval_logdir = os.path.join(evaldir, EVAL_SUBDIR)
        tf.gfile.MakeDirs(eval_logdir)

    evaluator = lib_evaluation.BaseEvaluator.make(
        unit, wmodel=wmodel, chronological=chronological)
    evaluator = lib_evaluation.EnsemblingEvaluator(evaluator, ensemble_size)

    if not sample_path and folder is None:
        raise ValueError(
            'Either --fold must be specified, or paths of npy files to load must '
            'be given, but not both.')
    if folder is not None:
        evaluate_fold(
            folder, evaluator, wmodel.hparams, eval_logdir, checkpoint_dir, index,
            unit, ensemble_size, chronological, data_dir)
    if sample_path is not None:
        evaluate_paths([sample_path], evaluator, wmodel.hparams,
                       eval_logdir, unit, ensemble_size, chronological)
    tf.logging.info('Done')


def evaluate_fold(fold, evaluator, hparams, eval_logdir, checkpoint_dir, index,
                  unit_in, ensemble_size, chronological, data_dir):
    """Writes to file the neg. loglikelihood of given fold (train/valid/test)."""
    eval_run_name = 'eval_%s_%s%s_%s_ensemble%s_chrono%s' % (
        lib_util.timestamp(), fold,
        '' if index is None else index, unit_in,
        ensemble_size, chronological)
    log_fname = '%s__%s.npz' % (os.path.basename(checkpoint_dir), eval_run_name)
    log_fpath = os.path.join(eval_logdir, log_fname)

    pianorolls = get_fold_pianorolls(fold, hparams, data_dir, index)

    rval = lib_evaluation.evaluate(evaluator, pianorolls)
    tf.logging.info('Writing to path: %s' % log_fpath)
    with lib_util.atomic_file(log_fpath) as p:
        np.savez_compressed(p, **rval)


def evaluate_paths(paths, evaluator, unused_hparams, eval_logdir, unit,
                   ensemble_size, chronological):
    """Evaluates negative loglikelihood of pianorolls from given paths."""
    for path in paths:
        name = 'eval_samples_%s_%s_ensemble%s_chrono%s' % (lib_util.timestamp(),
                                                           unit,
                                                           ensemble_size,
                                                           chronological)
        log_fname = '%s__%s.npz' % (os.path.splitext(os.path.basename(path))[0],
                                    name)
        log_fpath = os.path.join(eval_logdir, log_fname)

        pianorolls = get_path_pianorolls(path)
        rval = lib_evaluation.evaluate(evaluator, pianorolls)
        tf.logging.info('Writing evaluation statistics to %s', log_fpath)
        with lib_util.atomic_file(log_fpath) as p:
            np.savez_compressed(p, **rval)


def get_fold_pianorolls(fold, hparams, data_dir, index):
    dataset = lib_data.get_dataset(data_dir, hparams, fold)
    pianorolls = dataset.get_pianorolls()
    tf.logging.info('Retrieving pianorolls from %s set of %s dataset.',
                    fold, hparams.dataset)
    print_statistics(pianorolls)
    if index is not None:
        pianorolls = [pianorolls[int(index)]]
    return pianorolls


def get_path_pianorolls(path):
    pianoroll_fpath = os.path.join(tf.resource_loader.get_data_files_path(), path)
    tf.logging.info('Retrieving pianorolls from %s', pianoroll_fpath)
    with tf.gfile.Open(pianoroll_fpath, 'r') as p:
        pianorolls = np.load(p)
    if isinstance(pianorolls, np.ndarray):
        tf.logging.info(pianorolls.shape)
    print_statistics(pianorolls)
    return pianorolls


def print_statistics(pianorolls):
    """Prints statistics of given pianorolls, such as max and unique length."""
    if isinstance(pianorolls, np.ndarray):
        tf.logging.info(pianorolls.shape)
    tf.logging.info('# of total pieces in set: %d', len(pianorolls))
    lengths = [len(roll) for roll in pianorolls]
    if len(np.unique(lengths)) > 1:
        tf.logging.info('lengths %s', np.sort(lengths))
    tf.logging.info('max_len %d', max(lengths))
    tf.logging.info(
        'unique lengths %s',
        np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls)))
    tf.logging.info('shape %s', pianorolls[0].shape)


if __name__ == '__main__':
    import argparse

    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default=None,
                        help='Path to the base directory for different datasets.')
    parser.add_argument('eval_logdir', default=None,
                        help='Path to the base directory for saving evaluation '
                             'statistics.')
    parser.add_argument('fold', default=None,
                        help='Data fold on which to evaluate (valid or test)')
    parser.add_argument('fold_index', default=None,
                        help='Optionally, index of particular data point in fold to'
                             'evaluate.')
    parser.add_argument('unit', default=None, help='Note or frame or example.')
    parser.add_argument('ensemble_size', default=5,
                        help='Number of ensemble members to average.')
    parser.add_argument('chronological', default=False,
                        help='Indicates evaluation should proceed in chronological'
                             ' order.')
    parser.add_argument('checkpoint', default=None, help='Path to checkpoint directory.')
    parser.add_argument('sample_npy_path', default=None, help='Path to samples '
                                                              'to be evaluated.')

    args = parser.parse_args()
    main(args.checkpoint, args.eval_logdir, args.unit, args.chronological,
         args.ensemble_size, args.sample_npy_path, args.fold, args.fold_index,
         args.data_dir)
