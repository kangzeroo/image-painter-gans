import argparse
import tensorflow as tf
import os
from datetime import datetime

# enable eager execution......
# QUESTION - do we need to call this in model.py also for example???
tf.enable_eager_execution()
tf.executing_eagerly()

# NOTE -- for some fucking ass reason, need to change the import names when running local versus cloud............
#         local as in this modules README at least....
#
#         so lets wrap the import in a try catch
try:
    from model import ModelManager
    from generator import DataGenerator

except Exception as e:
    # # cloud - multi
    from trainer.model import ModelManager
    from trainer.generator import DataGenerator


def initialize_hyper_params(args_parser):

    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Args:
        args_parser
    """

    # Data files arguments
    args_parser.add_argument(
        '--job-name',
        help='Current gcloud job name',
        type=str,
    )
    args_parser.add_argument(
        '--train-batch-size',
        help='Batch size for each training step',
        type=int,
        default=1  # currently 25 throws memory errors...... NEED TO INCREASE THIS BABY (use 20 for now)
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""\
            Maximum number of training data epochs on which to train.
            If both --train-size and --num-epochs are specified,
            --train-steps will be: (train-size/train-batch-size) * num-epochs.\
            """,
        default=50,
        type=int,
    )
    args_parser.add_argument(
        '--gen-loss',
        help="""\
            The loss function for generator\
            * I dont think this is actually hooked up (should be implemented)
            """,
        default='mse',
        type=str,
    )
    args_parser.add_argument(
        '--disc-loss',
        help="""\
            The loss function for discriminator\
            * I dont think this is used.. (should be implemented)
            """,
        default='binary_crossentropy',
        type=str,
    )
    args_parser.add_argument(
        '--alpha',
        default=0.0004,
        type=float,
    )
    args_parser.add_argument(
        '--bucketname',
        default="lsun-roomsets",
        type=str,
    )
    args_parser.add_argument(
        '--staging-bucketname',
        default="theos_jobs",
        type=str,
    )
    args_parser.add_argument(
        '--epoch-save-frequency',
        default=1,
        type=int,
    )
    args_parser.add_argument(
        '--job-dir',
        # default="gs://temp/outputs",
        default="output_test_ckpt_1",
        type=str,
    )
    args_parser.add_argument(
        '--img-dir',
        # default="gs://temp/outputs",
        default="images/bedroom_val",
        type=str,
    )
    args_parser.add_argument(
        '--load-ckpt',
        default=False,
        type=bool,
        help="""\
            True or False specifying if to load the checkpoint
            file. If True, loads the highest epoch found in the
            ckpt folder in the output dir. If False, goes along
            as normal without loading any checkpoint"""
    )
    args_parser.add_argument(
        '--optimizer',
        default='AdadeltaOptimizer',
        help="""\
            The optimizer you want to use. Must be the same
            as in keras.optimizers"""
    )
    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help="Learning rate value for the optimizers - "
             "* I dont think this is used ( not needed with adadelta)",
        default=0.01,
        type=float
    )
    # Estimator arguments
    args_parser.add_argument(
        '--max-img-cnt',
        help="Number of maximum images to look at. Set to None if you"
             "want the whole dataset. Primarily used for testing purposes.",
        default=10,  # NOTE 300 imgs in validation set
        type=int
    )
    # Argument to turn on all logging
    args_parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    return args_parser.parse_args()


def main(params,
         global_shape=(256, 256, 3),
         local_shape=(128, 128, 3)):
    """
    TRAIN A BITCH
    :param params: dict - from argparser the paramaters of erting
    :param global_shape: tuple - assumed RGB - the shape inputted to the net
    :param local_shape: tuple - assumed RGB - local
    """

    # peace of mind
    print('Hyper-parameters:')
    print(params)

    # Set python level verbosity
    tf.logging.set_verbosity(params.verbosity)

    # Set C++ Graph Execution level verbosity  ------- dont know what this is
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[params.verbosity] / 10)

    # finally initialize the data generator
    train_datagen = DataGenerator(params, image_size=global_shape[:-1], local_size=local_shape[:-1])
    # next lets initialize our ModelManager (i.e. the thing that holds the GAN)
    mng = ModelManager(params)


    # Run the experiment
    time_start = datetime.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    # the actual call to run the experiment
    mng.run_training_procedure(train_datagen)

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("")


# not really sure the timming with this.... like I guess it is called on import ... ? Before __main__ obviously
argument_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialize_hyper_params(argument_parser)


if __name__ == '__main__':

    # we run the experiment with a single call
    main(HYPER_PARAMS)
