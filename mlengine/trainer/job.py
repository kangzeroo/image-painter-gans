import numpy as np
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input, Subtract, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.engine.network import Network
from keras.optimizers import Adadelta
import keras.backend as K
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from trainer.generator import createGenerator
from trainer.model import getBrains


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
BATCH_SIZE = 100
EPOCHS = 10
G_EPOCHS = int(EPOCHS * 0.2) # should be 90k on generator
D_EPOCHS = int(EPOCHS * 0.1) # should be 10k on discriminator
# BATCH_SIZE = 96
# EPOCHS = 500000
# G_EPOCHS = int(EPOCHS * 0.18) # should be 90k on generator
# D_EPOCHS = int(EPOCHS * 0.02) # should be 10k on discriminator

# Related to folder/file paths for input/output
BUCKET_NAME = 'lsun-roomsets'
INPUT_DIR = 'images/bedroom_val/'
JOB_DIR = 'jobs/'


def run_training_job(JOB_PARAMS, FILE_PARAMS, HYPER_PARAMS):
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
    train_datagen = createGenerator(BUCKET_NAME, INPUT_DIR, GLOBAL_SHAPE[:2], LOCAL_SHAPE[:2])
    dreamt_image = None

    # train over time
    for epoch in range(EPOCHS):
        # progress bar visualization (comment out in ML Engine)
        # progbar = generic_utils.Progbar(len(train_datagen))
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
                g_loss = gen_brain.train_on_batch([images, masks, mask_inv], valid)
            # train discriminator alone for 90k epochs
            # then train disc + gen for another 400k epochs. Total of 500k
            else:
                # throw in real unedited images with label VALID
                d_loss_real = disc_brain.train_on_batch([images, points], valid)
                # throw in A.I. generated images with label FAKE
                d_loss_fake = disc_brain.train_on_batch([generated_img, points], fake)
                # combine and set the disc loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if epoch >= G_EPOCHS + D_EPOCHS:
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

    if epoch // EPOCHS_PER_SAVED_WEIGHTS == 0:

        if dreamt_image is not None:
            OUTPUT_IMAGE_PATH = "gs://" + BUCKET_NAME + "/" + JOB_DIR + "output_images/epoch_" + epoch + "_image.png"
            with file_io.FileIO(OUTPUT_IMAGE_PATH, 'wb') as f:
                dreamt_image.save(f, "PNG")

        GEN_WEIGHTS_LOCAL_PATH = "output_models/epoch_" + epoch + "_generator.hdf5"
        DISC_WEIGHTS_LOCAL_PATH = "output_models/epoch_" + epoch + "_discriminator.hdf5"
        BRAIN_WEIGHTS_LOCAL_PATH = "output_models/epoch_" + epoch + "_brain.hdf5"

        gen_brain.save(GEN_WEIGHTS_LOCAL_PATH)
        copy_file_to_gcs(JOB_DIR, GEN_WEIGHTS_LOCAL_PATH)

        disc_brain.save(DISC_WEIGHTS_LOCAL_PATH)
        copy_file_to_gcs(JOB_DIR, DISC_WEIGHTS_LOCAL_PATH)

        brain.save(BRAIN_WEIGHTS_LOCAL_PATH)
        copy_file_to_gcs(JOB_DIR, BRAIN_WEIGHTS_LOCAL_PATH)



def copy_file_to_gcs(BUCKET_NAME, JOB_DIR, FILE_PATH):
  with file_io.FileIO(FILE_PATH, mode='rb') as input_f:
    with file_io.FileIO("gs://" + BUCKET_NAME + "/" + JOB_DIR + FILE_PATH, mode='w+') as output_f:
      output_f.write(input_f.read())
