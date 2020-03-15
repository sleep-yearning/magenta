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

"""Command line utility for exporting Coconet to SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_saved_model
from magenta.models.coconet import lib_tfsampling
import tensorflow.compat.v1 as tf


def main(checkpoint, destination, use_tf_sampling):
    model = None
    if use_tf_sampling:
        model = lib_tfsampling.CoconetSampleGraph(checkpoint)
        model.instantiate_sess_and_restore_checkpoint()
    else:
        model = lib_graph.load_checkpoint(checkpoint)
    tf.logging.info('Loaded graph.')
    lib_saved_model.export_saved_model(model, destination,
                                       [tf.saved_model.tag_constants.SERVING],
                                       use_tf_sampling)
    tf.logging.info('Exported SavedModel to %s.', destination)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', help='Path to the checkpoint to export.',
                        default=None)
    parser.add_argument('destination', help='Path to export SavedModel.',
                        default=None)
    parser.add_argument('--use_tf_sampling', help='Whether to export with'
                                                  'sampling in a TF while loop.',
                        default=True)
    args = parser.parse_args()
    main(args.checkpoint, args.destination, args.use_tf_sampling)
