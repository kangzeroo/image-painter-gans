import numpy as np
import argparse
import tensorflow as tf
import os
from datetime import datetime
from keras.utils import generic_utils
from PIL import Image
# import pdb  # run debugging from command
tf.enable_eager_execution()
tf.executing_eagerly()

# NOTE -- for some fucking ass reason, need to change the import names when running local versus cloud............
#         local as in this modules README at least....

# local
from model import ModelManager
from generator import DataGenerator
from utils import save_img, initialize_hyper_params

# # # cloud
# from trainer.model import full_disc_layer, ModelManager
# from trainer.generator import DataGenerator
# from trainer.utils import save_img, initialize_hyper_params


class Trainer:
    """
    puts everything together, defines the task (experiment) and also runs it
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

        # Directory to store output model and checkpoints
        model_dir = self.params.job_dir

        # If job_dir_reuse is False then remove the job_dir if it exists
        print("Resume training:", self.params.reuse_job_dir)
        if not self.params.reuse_job_dir:
            if tf.gfile.Exists(model_dir):
                tf.gfile.DeleteRecursively(model_dir)
                print("Deleted job_dir {} to avoid re-use".format(model_dir))
            else:
                print("No job_dir available to delete")
        else:
            print("Reusing job_dir {} if it exists".format(model_dir))

        # NOTE ---- i dont know what run_config does currently...
        self.run_config = tf.estimator.RunConfig(
            tf_random_seed=19830610,
            log_step_count_steps=1000,
            save_checkpoints_secs=120,  # change if you want to change frequency of saving checkpoints
            keep_checkpoint_max=3,
            model_dir=model_dir
        )
        run_config = self.run_config.replace(model_dir=model_dir)
        print("Model Directory:", run_config.model_dir)

        # finally initialize the data generator
        self.train_datagen = DataGenerator(params, image_size=global_shape[:-1], local_size=local_shape[:-1])
        # next lets initialize our ModelManager (i.e. the thing that holds the GAN)
        self.mng = ModelManager(
            params,
            global_image_tensor=(256, 256, 3),
            local_image_tensor=(128, 128, 3)
        )

    def run_task(self):
        """
        this defines the actual experiment
        :return:
        """
        self.mng.train(self.train_datagen)


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
        self.run_task()


        time_end = datetime.utcnow()
        print(".......................................")
        print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
        print("")
        time_elapsed = time_end - time_start
        print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
        print("")


# not really sure the timming with this.... like I guess it is called on import ... ? Before main obviously
args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialize_hyper_params(args_parser)


if __name__ == '__main__':

    # we run the experiment by initializing the trainer
    trainer = Trainer(
        params=HYPER_PARAMS
    )

    # then run
    trainer.main()
