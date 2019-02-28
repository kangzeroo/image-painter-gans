import numpy as np
import pdb  # run debugging from command
from keras import Input, Model
import argparse
import tensorflow as tf
import os
from datetime import datetime
from keras.utils import generic_utils
from PIL import Image
import pdb

# for some fucking ass reason, need to change the import names when running local versus cloud............

# # local
# from model import full_gen_layer, full_disc_layer
# from generator import DataGenerator
# from utils import save_img

# cloud
from trainer.model import full_gen_layer, full_disc_layer
from trainer.generator import DataGenerator
from trainer.utils import save_img


def initialise_hyper_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Args:
        args_parser
    """

    # Data files arguments
    args_parser.add_argument(
        '--train-batch-size',
        help='Batch size for each training step',
        type=int,
        default=20
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
        default="output",
        type=str,
    )
    args_parser.add_argument(
        '--img-dir',
        # default="gs://temp/outputs",
        default="images/bedroom_val",
        type=str,
    )
    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""\
            Flag to decide if the model checkpoint should
            be re-used from the job-dir. If False then the
            job-dir will be deleted"""
    )
    args_parser.add_argument(
        '--optimizer',
        default='Adadelta',
        help="""\
            The optimizer you want to use. Must be the same
            as in keras.optimizers"""
    )
    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help="Learning rate value for the optimizers",
        default=0.1,
        type=float
    )
    # Estimator arguments
    args_parser.add_argument(
        '--max-img-cnt',
        help="Number of maximum images to look at. Set to None if you"
             "want the whole dataset. Primarily used for testing purposes.",
        default=500,
        type=int
    )
    args_parser.add_argument(
        '--run-type',
        help="Number of maximum images to look at. Set to None if you"
             "want the whole dataset.",
        default='default',  # can be default - or local --- which images to look at. Hack for me to work in local with local images
        type=str
    )
    args_parser.add_argument(
        '--local-img-dir',
        help="local directory of images. only used when run-type is local",
        default='../temp',  # can be default - or local --- which images to look at. Hack for me to work in local with local images
        type=str
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


def run_experiment(run_config):
    """
    trains the gan by looping through epochs manually
    :param run_config:
    :return:
    """

    full_img = Input(shape=global_shape)
    erased_img = Input(shape=global_shape)
    # clip_img = Input(shape=local_shape)
    mask = Input(shape=(global_shape[0], global_shape[1], 1))
    # ones = Input(shape=(global_shape[0], global_shape[1], 1))
    clip_coords = Input(shape=(4,), dtype='int32')

    gen_brain, gen_model = full_gen_layer(
        params=HYPER_PARAMS,
        full_img=full_img,
        mask=mask,
        erased_image=erased_img
    )

    disc_brain, disc_model = full_disc_layer(
        params=HYPER_PARAMS,
        global_shape=global_shape,
        local_shape=local_shape,
        full_img=full_img,
        clip_coords=clip_coords)

    # the final brain
    disc_model.trainable = False
    connected_disc = Model(inputs=[full_img, clip_coords], outputs=disc_model)
    connected_disc.name = 'Connected-Discrimi-Hater'

    brain = Model(inputs=[full_img, mask, erased_img, clip_coords],
                  outputs=[gen_model, connected_disc([gen_model, clip_coords])])
    brain.compile(loss=['mse', 'binary_crossentropy'],
                  loss_weights=[1.0, HYPER_PARAMS.alpha], optimizer=HYPER_PARAMS.optimizer)

    if HYPER_PARAMS.verbosity == 'INFO':
        print(gen_brain)
        print(disc_brain)

        print(gen_model)
        print(disc_model)
        print(connected_disc)
        brain.summary()
        # view_models(brain, '../summaries/brain.png')

    # data generator
    train_datagen = DataGenerator(HYPER_PARAMS, image_size=global_shape[:-1], local_size=local_shape[:-1])
    batch_count = 0
    # train over time
    dreamt_image = None
    # g_epochs = int(HYPER_PARAMS.num_epochs * 0.18)
    # d_epochs = int(HYPER_PARAMS.num_epochs * 0.02)
    g_epochs = 5
    d_epochs = 5
    for epoch in range(HYPER_PARAMS.num_epochs):
        print('\nstarting epoch {}\n'.format(epoch))
        # progress bar visualization (comment out in ML Engine)
        progbar = generic_utils.Progbar(len(train_datagen))
        for images, points, masks in train_datagen.flow(batch_size=HYPER_PARAMS.train_batch_size):
            masks_inv = 1 - masks
            erased_imgs = images * masks_inv
            # generate the inputs (images)
            generated_img = gen_brain.predict([images, masks_inv, erased_imgs])
            # generate the labels
            valid = np.ones((HYPER_PARAMS.train_batch_size, 1))
            fake = np.zeros((HYPER_PARAMS.train_batch_size, 1))
            # the gen and disc losses
            g_loss = 0.0
            d_loss = 0.0
            # we must train the neural nets seperately, and then together
            # train generator for 90k epochs
            if epoch < g_epochs:
                # set the gen loss
                g_loss = gen_brain.train_on_batch([images, masks_inv, erased_imgs], generated_img)
            # train discriminator alone for 90k epochs
            # then train disc + gen for another 400k epochs. Total of 500k
            else:
                # throw in real unedited images with label VALID
                d_loss_real = disc_brain.train_on_batch([images, points], valid)
                # throw in A.I. generated images with label FAKE
                d_loss_fake = disc_brain.train_on_batch([generated_img, points], fake)
                # combine and set the disc loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if epoch >= g_epochs + d_epochs:
                    # train the entire brain
                    g_loss = brain.train_on_batch([images, masks, erased_imgs, points], [images, valid])
                    # and update the generator loss
                    g_loss = g_loss[0] + HYPER_PARAMS.alpha * g_loss[1]
            # progress bar visualization (comment out in ML Engine)
            progbar.add(images.shape[0], values=[("Disc Loss: ", d_loss), ("Gen mse: ", g_loss)])
            batch_count += 1
            # save the generated image
            last_img = generated_img[0]
            last_img *= 255
            dreamt_image = Image.fromarray(last_img.astype('uint8'), 'RGB')

        # gen_brain.save(f"./outputs/models/batch_{batch_count}_generator.h5")
        # disc_brain.save(f"./outputs/models/batch_{batch_count}discriminator.h5")

        if epoch % HYPER_PARAMS.epoch_save_frequency == 0 and epoch > 0:
            if dreamt_image is not None:
                OUTPUT_IMAGE_PATH = 'gs://{}/{}/images/epoch_{}_image.png'.format(HYPER_PARAMS.staging_bucketname, HYPER_PARAMS.job_dir, epoch)
                save_img(save_path=OUTPUT_IMAGE_PATH, img_data=dreamt_image)

            # GEN_WEIGHTS_LOCAL_PATH = "models/epoch_" + str(epoch) + "_generator.hdf5"
            # DISC_WEIGHTS_LOCAL_PATH = "models/epoch_" + str(epoch) + "_discriminator.hdf5"
            # BRAIN_WEIGHTS_LOCAL_PATH = "models/epoch_" + str(epoch) + "_brain.hdf5"

            # GEN_WEIGHTS_LOCAL_PATH = "{}/output_models/epoch_{}_generator.hdf5".format(HYPER_PARAMS.staging_bucketname, str(epoch))
            # DISC_WEIGHTS_LOCAL_PATH = "{}/output_models/epoch_{}_discriminator.hdf5".format(HYPER_PARAMS.staging_bucketname, str(epoch))
            # BRAIN_WEIGHTS_LOCAL_PATH = "{}/output_models/epoch_{}_brain.hdf5".format(HYPER_PARAMS.staging_bucketname, str(epoch))

            # GEN_WEIGHTS_LOCAL_PATH = "https://console.cloud.google.com/storage/browser/{}/output_models/epoch_{}_generator.hdf5".format(HYPER_PARAMS.job_dir, str(epoch))
            # DISC_WEIGHTS_LOCAL_PATH = "https://console.cloud.google.com/storage/browser/{}/output_models/epoch_{}_discriminator.hdf5".format(HYPER_PARAMS.job_dir, str(epoch))
            # BRAIN_WEIGHTS_LOCAL_PATH = "https://console.cloud.google.com/storage/browser/{}/output_models/epoch_{}_brain.hdf5".format(HYPER_PARAMS.job_dir, str(epoch))

            # gen_brain.save(GEN_WEIGHTS_LOCAL_PATH)
            # copy_file_to_gcs(HYPER_PARAMS.staging_bucketname, HYPER_PARAMS.job_dir, GEN_WEIGHTS_LOCAL_PATH)

            # disc_brain.save(DISC_WEIGHTS_LOCAL_PATH)
            # # copy_file_to_gcs(HYPER_PARAMS.staging_bucketnamev, DISC_WEIGHTS_LOCAL_PATH)
            #
            # brain.save(BRAIN_WEIGHTS_LOCAL_PATH)
            # # copy_file_to_gcs(HYPER_PARAMS.staging_bucketnamev, BRAIN_WEIGHTS_LOCAL_PATH)


def main():
    """
    main driver of the task.
    :return:
    """
    print('')
    print('Hyper-parameters:')
    print(HYPER_PARAMS)
    print('')

    # Set python level verbosity
    tf.logging.set_verbosity(HYPER_PARAMS.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[HYPER_PARAMS.verbosity] / 10)

    # Directory to store output model and checkpoints
    model_dir = HYPER_PARAMS.job_dir

    # If job_dir_reuse is False then remove the job_dir if it exists
    print("Resume training:", HYPER_PARAMS.reuse_job_dir)
    if not HYPER_PARAMS.reuse_job_dir:
        if tf.gfile.Exists(model_dir):
            tf.gfile.DeleteRecursively(model_dir)
            print("Deleted job_dir {} to avoid re-use".format(model_dir))
        else:
            print("No job_dir available to delete")
    else:
        print("Reusing job_dir {} if it exists".format(model_dir))

    run_config = tf.estimator.RunConfig(
        tf_random_seed=19830610,
        log_step_count_steps=1000,
        save_checkpoints_secs=120,  # change if you want to change frequency of saving checkpoints
        keep_checkpoint_max=3,
        model_dir=model_dir
    )

    run_config = run_config.replace(model_dir=model_dir)

    print("Model Directory:", run_config.model_dir)

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")
    run_experiment(run_config)

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("")


args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)


if __name__ == '__main__':

    global_shape = (256, 256, 3)
    local_shape = (128, 128, 3)

    # hyperparameters
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)

    main()
