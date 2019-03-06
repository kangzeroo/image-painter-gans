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
    # local call ---
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

    Is this ok in utils?

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
        default=2  # currently 25 throws memory errors...... NEED TO INCREASE THIS BABY (use 20 for now)
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""\
            Maximum number of training data epochs on which to train.
            If both --train-size and --num-epochs are specified,
            --train-steps will be: (train-size/train-batch-size) * num-epochs.\
            """,
        default=500,
        type=int,
    )
    args_parser.add_argument(
        '--gen-loss',
        help="""\
            The loss function for generator\
            """,
        default='mse',
        type=str,
    )
    args_parser.add_argument(
        '--disc-loss',
        help="""\
            The loss function for discriminator\
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
        default=2,
        type=int,
    )
    args_parser.add_argument(
        '--job-dir',
        # default="gs://temp/outputs",
        default="output_test_ckpt",
        type=str,
    )
    args_parser.add_argument(
        '--img-dir',
        # default="gs://temp/outputs",
        default="images/bedroom_train",
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
        help="Learning rate value for the optimizers",
        default=0.01,
        type=float
    )
    # Estimator arguments
    args_parser.add_argument(
        '--max-img-cnt',
        help="Number of maximum images to look at. Set to None if you"
             "want the whole dataset. Primarily used for testing purposes.",
        default=2,
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


class Trainer:
    """
    puts everything together and runs the training task from modelmanager
    """
    def __init__(
        self,
        params,
        global_shape=(256, 256, 3),
        local_shape=(128, 128, 3),
    ):
        """
        trainer - manages and runs our experiment. holds everything related to the experiment (task) and runs the
        experiment. I.e. holds the generator, the modelmanager, the paramaters etc

        initialization will create the GAN (in mng) and the data generator

        :param params: dict - from argparser the paramaters of erting
        :param global_shape: tuple - assumed RGB - the shape inputted to the net
        :param local_shape: tuple - assumed RGB - local
        """
        # set up
        print('initializing task')
        self.params = params

        # peace of mind
        print('Hyper-parameters:')
        print(self.params)

        # Set python level verbosity
        tf.logging.set_verbosity(self.params.verbosity)

        # Set C++ Graph Execution level verbosity  ------- dont know what this is
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[self.params.verbosity] / 10)

        # finally initialize the data generator
        self.train_datagen = DataGenerator(params, image_size=global_shape[:-1], local_size=local_shape[:-1])
        # next lets initialize our ModelManager (i.e. the thing that holds the GAN)
        self.mng = ModelManager(
            params,
        )

    def main(self):
        """
        runs the experiment baby
        :return:
        """

        # Run the experiment
        time_start = datetime.utcnow()
        print("")
        print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
        print(".......................................")

        # the actual call to run the experiment
        self.mng.run_training_procedure(self.train_datagen)

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

    # we run the experiment by initializing the trainer
    trainer = Trainer(
        params=HYPER_PARAMS
    )

    # then run
    trainer.main()
