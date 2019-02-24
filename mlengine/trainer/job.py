import numpy as np
import pdb
import os
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input, Subtract, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.engine.network import Network
from keras.optimizers import Adadelta
import keras.backend as K
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io import file_io

from trainer.generator import createGenerator
from trainer.model import getBrains

# Debug in python interactive terminal
# insert the below line anywhere in your code
# pdb.set_trace()

# Related to job run
EPOCHS_PER_SAVED_WEIGHTS = 1

# Related to model
GLOBAL_SHAPE = (256,256,3)
LOCAL_SHAPE = (128,128,3)
OPTIMIZER = Adadelta()
FULL_IMG = Input(shape=GLOBAL_SHAPE)
MASK = Input(shape=(GLOBAL_SHAPE[0], GLOBAL_SHAPE[1], 1))
ONES = Input(shape=(GLOBAL_SHAPE[0], GLOBAL_SHAPE[1], 1))
CLIP_COORDS = Input(shape=(4,), dtype='int32')
HOLE_MIN=64
HOLE_MAX=128

# Related to Hyper-Params
ALPHA = 0.0004
BATCH_SIZE = 1
EPOCHS = 2
G_EPOCHS = int(EPOCHS * 0.2) # should be 90k on generator
D_EPOCHS = int(EPOCHS * 0.1) # should be 10k on discriminator
MAX_TRAINING_IMAGES = 3
# BATCH_SIZE = 96
# EPOCHS = 500000
# G_EPOCHS = int(EPOCHS * 0.18) # should be 90k on generator
# D_EPOCHS = int(EPOCHS * 0.02) # should be 10k on discriminator
# MAX_TRAINING_IMAGES = 3000000

# Related to folder/file paths for input/output
BUCKET_NAME = 'lsun-roomsets'
INPUT_DIR = 'images/bedroom_val/'
JOB_DIR = 'jobs/'


def run_training_job(JOB_PARAMS, FILE_PARAMS, HYPER_PARAMS, TIMESTAMP):

    writeTestPath = "gs://" + BUCKET_NAME + "/" + JOB_DIR + "output_" + TIMESTAMP + ".txt"
    with file_io.FileIO(writeTestPath, mode='wb+') as of:
        of.write("Write test passed for " + writeTestPath)
        print("Testing GCS write permission. Wrote file " + writeTestPath)

    # get the brains
    brain, gen_brain, disc_brain = getBrains(
        GLOBAL_SHAPE,
        LOCAL_SHAPE,
        OPTIMIZER,

        FULL_IMG,
        MASK,
        ONES,
        CLIP_COORDS,

        ALPHA,

        BUCKET_NAME,
        JOB_DIR,
    )
    # get the data generator
    train_datagen = createGenerator(BUCKET_NAME, INPUT_DIR, GLOBAL_SHAPE[:2], LOCAL_SHAPE[:2], MAX_TRAINING_IMAGES)
    dreamt_image = None

    # train over time
    for epoch in range(EPOCHS):
        # progress bar visualization (comment out in ML Engine)
        # progbar = generic_utils.Progbar(len(train_datagen))
        print('epoch ' + str(epoch) + ' ----- processing batches from .flow()')
        for images, points, masks in train_datagen.flow(BATCH_SIZE, BUCKET_NAME, HOLE_MIN, HOLE_MAX):
            # and the matrix of ones that we depend on in the neural net to inverse masks
            mask_inv = np.ones((len(images), GLOBAL_SHAPE[0], GLOBAL_SHAPE[1], 1))
            # generate the inputs (images)
            generated_img = gen_brain.predict([images, masks, mask_inv])
            # generate the labels
            valid = np.ones((BATCH_SIZE, 1))
            fake = np.zeros((BATCH_SIZE, 1))
            # the gen and disc losses
            g_loss = 0.0
            d_loss = 0.0

            # we must train the neural nets seperately, and then together
            # train generator for 90k epochs
            if epoch < G_EPOCHS:
                # set the gen loss
                print('training gen.net loss...')
                g_loss = gen_brain.train_on_batch([images, masks, mask_inv], generated_img)
            # train discriminator alone for 90k epochs
            # then train disc + gen for another 400k epochs. Total of 500k
            else:
                print('training disc.net loss...')
                # throw in real unedited images with label VALID
                d_loss_real = disc_brain.train_on_batch([images, points], valid)
                # throw in A.I. generated images with label FAKE
                d_loss_fake = disc_brain.train_on_batch([generated_img, points], fake)
                # combine and set the disc loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if epoch >= G_EPOCHS + D_EPOCHS:
                    print('training gans.net combined loss...')
                    # train the entire brain
                    g_loss = brain.train_on_batch([images, masks, mask_inv, points], [images, valid])
                    # and update the generator loss
                    g_loss = g_loss[0] + ALPHA * g_loss[1]
            # progress bar visualization (comment out in ML Engine)
            # progbar.add(images.shape[0], values=[("Disc Loss: ", d_loss), ("Gen mse: ", g_loss)])
            # save the generated image
            last_img = generated_img[0]
            last_img[:,:,0] = last_img[:,:,0]*255
            last_img[:,:,1] = last_img[:,:,1]*255
            last_img[:,:,2] = last_img[:,:,2]*255
            dreamt_image = Image.fromarray(last_img.astype(int), 'RGB')

        print('--- End of Epoch ' + str(epoch))
        if epoch // EPOCHS_PER_SAVED_WEIGHTS == 0:

            OUTPUT_IMAGE_PATH = "gs://" + BUCKET_NAME + "/" + JOB_DIR + "output_images/epoch_" + str(epoch) + "_image.png"
            print("Saving last generated image to " + OUTPUT_IMAGE_PATH)
            if dreamt_image is not None:
                with file_io.FileIO(OUTPUT_IMAGE_PATH, 'wb') as f:
                    dreamt_image.save(f, "PNG")

            # local testing
            GEN_WEIGHTS_LOCAL_PATH = "/output_models/epoch_" + str(epoch) + "_generator.h5"
            DISC_WEIGHTS_LOCAL_PATH = "/output_models/epoch_" + str(epoch) + "_discriminator.h5"
            BRAIN_WEIGHTS_LOCAL_PATH = "/output_models/epoch_" + str(epoch) + "_brain.h5"

            GEN_WEIGHTS_GCS_PATH = "gs://" + BUCKET_NAME + "/" + JOB_DIR + "output_models/epoch_" + str(epoch) + "_generator.h5"
            DISC_WEIGHTS_GCS_PATH = "gs://" + BUCKET_NAME + "/" + JOB_DIR + "output_models/epoch_" + str(epoch) + "_discriminator.h5"
            BRAIN_WEIGHTS_GCS_PATH = "gs://" + BUCKET_NAME + "/" + JOB_DIR + "output_models/epoch_" + str(epoch) + "_brain.h5"

            print('Saving weights to local path ' + GEN_WEIGHTS_LOCAL_PATH)
            gen_brain.save(GEN_WEIGHTS_LOCAL_PATH)
            print('Copying weights to GCS ' + GEN_WEIGHTS_GCS_PATH)
            copy_file_to_gcs(GEN_WEIGHTS_LOCAL_PATH, GEN_WEIGHTS_GCS_PATH)

            print('Saving weights to local path ' + DISC_WEIGHTS_LOCAL_PATH)
            disc_brain.save(DISC_WEIGHTS_LOCAL_PATH)
            print('Copying weights to GCS ' + DISC_WEIGHTS_GCS_PATH)
            copy_file_to_gcs(DISC_WEIGHTS_GCS_PATH, DISC_WEIGHTS_GCS_PATH)

            print('Saving weights to local path ' + BRAIN_WEIGHTS_LOCAL_PATH)
            brain.save(BRAIN_WEIGHTS_LOCAL_PATH)
            print('Copying weights to GCS ' + BRAIN_WEIGHTS_GCS_PATH)
            copy_file_to_gcs(BRAIN_WEIGHTS_LOCAL_PATH, BRAIN_WEIGHTS_GCS_PATH)

            # tf checkpoints
            # https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restore

            # load checkpoints
            # model_path_glob = 'checkpoint.*'
            # if not self.job_dir.startswith('gs://'):
            #     model_path_glob = os.path.join(self.job_dir, model_path_glob)
            # checkpoints = glob.glob(model_path_glob)
            # if len(checkpoints) > 0:
            #     checkpoints.sort()
            #     brain = load_model(checkpoints[-1])

            # save checkpoints
            # checkpoint = tf.train.Checkpoint(optimizer=OPTIMIZER, model=brain)
            # checkpoint.write(
            #     'gs://bucketpath/checkpoints',
            #     session=None
            # )

            # save tensorboard stats
            # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard#class_tensorboard




def copy_file_to_gcs(LOCAL_PATH, GCS_PATH):
  with file_io.FileIO(LOCAL_PATH, mode='rb') as input_f:
    with file_io.FileIO(GCS_PATH, mode='wb+') as output_f:
      output_f.write(input_f.read())