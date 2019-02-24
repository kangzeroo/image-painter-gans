import argparse
import glob
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model

from tensorflow.python.lib.io import file_io

import trainer.model as model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train-files',
      nargs='+',
      help='Training folder, local or GCS',
      default='gs://lsun-roomsets/images/bedroom_train/')
  parser.add_argument(
      '--eval-files',
      nargs='+',
      help='Evaluation folder, local or GCS',
      default='gs://lsun-roomsets/images/bedroom_val/')
  parser.add_argument(
      '--job-dir',
      type=str,
      help='GCS or local dir to write checkpoints and export model',
      default='gs://lsun-roomsets/progress/')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=500000,
      help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=2000,
      type=int)
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=40,
      help='Batch size for training steps')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=40,
      help='Batch size for evaluation steps')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.003,
      help='Learning rate for SGD')
  parser.add_argument(
      '--eval-frequency',
      default=10,
      help='Perform one evaluation per n epochs')
  parser.add_argument(
      '--first-layer-size',
      type=int,
      default=256,
      help='Number of nodes in the first layer of DNN')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=2,
      help='Number of layers in DNN')
  parser.add_argument(
      '--scale-factor',
      type=float,
      default=0.25,
      help="""Rate of decay size of layer for Deep Neural Net.
        max(2, int(first_layer_size * scale_factor**i))""")
  parser.add_argument(
      '--eval-num-epochs',
      type=int,
      default=1,
      help='Number of epochs during evaluation')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=20,
      help='Maximum number of epochs on which to train')
  parser.add_argument(
      '--checkpoint-epochs',
      type=int,
      default=5,
      help='Checkpoint per n training epochs')

  args, _ = parser.parse_known_args()
  train_and_evaluate(args)
