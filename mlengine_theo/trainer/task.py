import numpy as np
import argparse
import tensorflow as tf
import os
from datetime import datetime
from keras.utils import generic_utils
from PIL import Image
# import pdb  # run debugging from command


# NOTE -- for some fucking ass reason, need to change the import names when running local versus cloud............
#         local as in this modules README at least....

# local
# from model import full_gen_layer, full_disc_layer
# from generator import DataGenerator
# from utils import save_img

# # cloud
from trainer.model import full_gen_layer, full_disc_layer, ModelManager
from trainer.generator import DataGenerator
from trainer.utils import save_img, initialize_hyper_params


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
            global_shape=(256, 256, 3),
            local_shape=(128, 128, 3)
        )

    def run_task(self):
        """
        this defines the actual experiment
        :return:
        """
        # data generator
        batch_count = 0
        # train over time
        dreamt_image = None
        # g_epochs = int(self.params.num_epochs * 0.18)
        # d_epochs = int(self.params.num_epochs * 0.02)
        g_epochs = 2
        d_epochs = 2
        for epoch in range(self.params.num_epochs):
            print('\nstarting epoch {}\n'.format(epoch))
            # progress bar visualization (comment out in ML Engine)
            progbar = generic_utils.Progbar(len(self.train_datagen))
            for images, points, masks in self.train_datagen.flow(batch_size=self.params.train_batch_size):
                masks_inv = 1 - masks
                erased_imgs = images * masks_inv
                # generate the inputs (images)
                generated_img = self.mng.gen_brain.predict([images, masks_inv, erased_imgs])
                # generate the labels
                valid = np.ones((self.params.train_batch_size, 1))
                fake = np.zeros((self.params.train_batch_size, 1))
                # the gen and disc losses
                g_loss = 0.0
                d_loss = 0.0
                # we must train the neural nets seperately, and then together
                # train generator for 90k epochs
                if epoch < g_epochs:
                    # set the gen loss
                    g_loss = self.mng.gen_brain.train_on_batch([images, masks_inv, erased_imgs], generated_img)
                # train discriminator alone for 90k epochs
                # then train disc + gen for another 400k epochs. Total of 500k
                else:
                    # throw in real unedited images with label VALID
                    d_loss_real = self.mng.disc_brain.train_on_batch([images, points], valid)
                    # throw in A.I. generated images with label FAKE
                    d_loss_fake = self.mng.disc_brain.train_on_batch([generated_img, points], fake)
                    # combine and set the disc loss
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    if epoch >= g_epochs + d_epochs:
                        # train the entire brain
                        g_loss = self.mng.brain.train_on_batch([images, masks, erased_imgs, points], [images, valid])
                        # and update the generator loss
                        g_loss = g_loss[0] + self.params.alpha * g_loss[1]
                # progress bar visualization (comment out in ML Engine)
                progbar.add(images.shape[0], values=[("Disc Loss: ", d_loss), ("Gen mse: ", g_loss)])
                batch_count += 1
                # save the generated image
                last_img = generated_img[0]
                last_img *= 255
                dreamt_image = Image.fromarray(last_img.astype('uint8'), 'RGB')

            # clean this up ......... !!! !!!! !!!!! ! ! !! ! ! !! !!! ! ! !!
            # gen_brain.save(f"./outputs/models/batch_{batch_count}_generator.h5")
            # disc_brain.save(f"./outputs/models/batch_{batch_count}discriminator.h5")
            if epoch % self.params.epoch_save_frequency == 0 and epoch > 0:
                if dreamt_image is not None:
                    output_image_path = 'gs://{}/{}/images/epoch_{}_image.png'.format(
                        self.params.staging_bucketname,
                        self.params.job_dir, epoch
                    )
                    save_img(save_path=output_image_path, img_data=dreamt_image)

                # GEN_WEIGHTS_LOCAL_PATH = "models/epoch_" + str(epoch) + "_generator.hdf5"
                # DISC_WEIGHTS_LOCAL_PATH = "models/epoch_" + str(epoch) + "_discriminator.hdf5"
                # BRAIN_WEIGHTS_LOCAL_PATH = "models/epoch_" + str(epoch) + "_brain.hdf5"

                # GEN_WEIGHTS_LOCAL_PATH = "{}/output_models/epoch_{}_generator.hdf5".format(self.params.staging_bucketname, str(epoch))
                # DISC_WEIGHTS_LOCAL_PATH = "{}/output_models/epoch_{}_discriminator.hdf5".format(self.params.staging_bucketname, str(epoch))
                # BRAIN_WEIGHTS_LOCAL_PATH = "{}/output_models/epoch_{}_brain.hdf5".format(self.params.staging_bucketname, str(epoch))

                # GEN_WEIGHTS_LOCAL_PATH = "https://console.cloud.google.com/storage/browser/{}/output_models/epoch_{}_generator.hdf5".format(HYPER_PARAMS.job_dir, str(epoch))
                # DISC_WEIGHTS_LOCAL_PATH = "https://console.cloud.google.com/storage/browser/{}/output_models/epoch_{}_discriminator.hdf5".format(HYPER_PARAMS.job_dir, str(epoch))
                # BRAIN_WEIGHTS_LOCAL_PATH = "https://console.cloud.google.com/storage/browser/{}/output_models/epoch_{}_brain.hdf5".format(HYPER_PARAMS.job_dir, str(epoch))

                # gen_brain.save(GEN_WEIGHTS_LOCAL_PATH)
                # copy_file_to_gcs(self.params.staging_bucketname, self.params.job_dir, GEN_WEIGHTS_LOCAL_PATH)

                # disc_brain.save(DISC_WEIGHTS_LOCAL_PATH)
                # # copy_file_to_gcs(self.params.staging_bucketnamev, DISC_WEIGHTS_LOCAL_PATH)
                #
                # brain.save(BRAIN_WEIGHTS_LOCAL_PATH)
                # # copy_file_to_gcs(self.params.staging_bucketnamev, BRAIN_WEIGHTS_LOCAL_PATH)

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
