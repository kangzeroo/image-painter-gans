import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Subtract, Add, Multiply
from tensorflow.keras.layers import BatchNormalization
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import Model
import tensorflow.train as optimizers
import tensorflow.keras.backend as k
from keras.utils import generic_utils
import pdb  # for debugging
import numpy as np
from PIL import Image

try:
    from utils import save_img
except Exception as e:
    from trainer.utils import save_img


def model_generator():
    """
    creates the generator
    :param input_shape:
    :return:
    """
    # in_layer = tfe.Variable(global_image_tensor)

    model_gen = tf.keras.Sequential([
        Conv2D(64, kernel_size=5, strides=1, padding='same',
               dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, strides=2,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=2,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(2, 2)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(4, 4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(8, 8)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(16, 16)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(128, kernel_size=4, strides=2,
                        padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2DTranspose(64, kernel_size=4, strides=2,
                        padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('sigmoid'),
        Conv2D(3, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('sigmoid')
    ])

    # model_gen = Model(inputs=in_layer, outputs=model, name='Gener8tor')
    return model_gen


def model_discriminator():
    # def crop_image(img, crop):
    #     return tf.image.crop_to_bounding_box(img,
    #                                          crop.shape[1],
    #                                          crop.shape[0],
    #                                          crop.shape[3] - crop.shape[1],
    #                                          crop.shape[2] - crop.shape[0])
    #
    # in_pts = tfe.Variable((4,), dtype='int32')
    # cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
    #                   output_shape=local_image_tensor)
    # g_img = tfe.Variable(global_shape)
    # # l_img = cropping([g_img, in_pts])
    # l_img = tf.image.crop_to_bounding_box(g_img,
    #                                      clip_coords[1],
    #                                      clip_coords[0],
    #                                      clip_coords[3] - clip_coords[1],
    #                                      clip_coords[2] - clip_coords[0])

    x_l = tf.keras.Sequential([
        Conv2D(64, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Flatten(),
        Dense(1024, activation='relu')
    ])

    x_g = tf.keras.Sequential([
        Conv2D(64, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Flatten(),
        Dense(1024, activation='relu')
    ])

    return x_l, x_g

# # i thought maybe we would train the whole gan by making another model class.... but settled to do it on the individual
# # models instead ...
#
# class FullGAN(Model):
#     def __init__(self, disc_model, generator_model):
#         super(FullGAN, self).__init__()
#         self.disc_model, self.generator_model = disc_model, generator_model
#
#     def call(self, erased_imgs, roi_imgs):
#         """
#         we put the generator and the discriminator together and train the hole ting babygirl
#         :param imgs: tensor of images --- typically output from generator
#         :param cropped_imgs: the generated images by mask......
#         :return:
#         """
#         x_gen = self.generator_model(erased_imgs)
#         x_disc = self.disc_model(x_gen, roi_imgs)
#
#         return x_gen, x_disc


class DiscConnected(Model):
    def __init__(self):
        super(DiscConnected, self).__init__()
        self.disc_model_local, self.disc_model_global = model_discriminator()

    def call(self, inputs, cropped_imgs, training=False):
        # x = tf.keras.cont(inputs)
        x = tf.concat([self.disc_model_local(cropped_imgs), self.disc_model_global(inputs)], axis=1)
        x = Dense(1, activation='sigmoid')(x)
        return x


class ModelManager(Model):
    """

    this class holds all the components for the GAN and all the methods you'd want baby.

    Made up of:
        generator - generates images
        discriminator - local and global branches

    """
    def __init__(
        self,
        params,
    ):
        """
        manage the gan. we construct the discriminator and generator on initialization....... Note, these are not "compiled"
        yet, since we are using eager, they only get compiled when called
        :param params: DICT - from argparser
        """
        super(ModelManager, self).__init__()
        self.params = params
        self.gen_loss_history, self.disc_loss_history, self.brain_history = [], [], []
        self.gen_optimizer = getattr(tf.train, self.params.optimizer)(learning_rate=self.params.learning_rate)  # NOTE THIS might be different from paper
        self.disc_optimizer = getattr(tf.train, self.params.optimizer)(learning_rate=self.params.learning_rate)  # NOTE THIS might be different from paper
        # self.brain_optimizer = getattr(tf.train, self.params.optimizer)(learning_rate=self.params.learning_rate)  # NOTE THIS might be different from paper

        # this is the generator model
        self.gen_model = model_generator()

        # full discriminator (i.e. global + local branch)
        self.disc_model = DiscConnected()

        # this is the full brain (generator + discriminator)
        # self.full_brain = FullGAN(disc_model=self.disc_model, generator_model=self.gen_model)

        # if params.verbosity == 'INFO':
        #     print(disc_model)
        #     disc_brain.summary()
        #     # view_models(disc_brain, '../summaries/disc_brain.png')

    def predict_gen(self, img):
        """
        predict the generator on a given image
        :param erased_imgs:
        :return:
        """

        generated_img = self.gen_model(img)

        return generated_img

    def train_disc(self, imgs, masked_imgs, labels):
        """
        we kinda need to create it on the fly also
        :param imgs:
        :param masked_imgs:
        :param labels:
        :return:
        """

        with tf.GradientTape() as tape:
            output = self.disc_model.call(imgs, masked_imgs)
            loss_value = tf.losses.sigmoid_cross_entropy(labels, output)  # we use simoid_cross_entropy in replace of keras' binary cross entropy

        self.disc_loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, self.disc_model.trainable_variables)  # this takes a long time on cpu
        self.disc_optimizer.apply_gradients(
            zip(
                grads,
                self.disc_model.trainable_variables
            ),
            global_step=tf.train.get_or_create_global_step()
        )

        return loss_value

    def train_full_brain(self, erased_imgs, images, roi_imgs, valid):
        """
        trains the entire brain (i.e. gen + disc). can make probably write this more general and combine the other 2 training methods....
        :param erased_imgs:
        :param images:
        :param roi_imgs:
        :param valid:
        :return:
        """
        with tf.GradientTape() as tape:
            # output_gen, output_disc = self.full_brain.call(erased_imgs, roi_imgs)
            output_gen = self.gen_model(erased_imgs)
            loss_value_gen = tf.losses.mean_squared_error(
                images,
                output_gen,
                weights=1.0
            )

        # im not sure how to fully train the entire model....
        # we will first just try to train the generator then the discriminator....
        # train the generator
        grads_gen = tape.gradient(loss_value_gen, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(
                grads_gen,
                self.gen_model.trainable_variables,
            ),
            global_step=tf.train.get_or_create_global_step()
        )

        # now lets teach the discriminator...
        with tf.GradientTape() as tape:
            output_disc = self.disc_model.call(images, roi_imgs)
            loss_value_disc = tf.losses.sigmoid_cross_entropy(   # we use simoid_cross_entropy in replace of keras' binary cross entropy
                valid,
                output_disc,
                weights=self.params.alpha
            )

        grads_disc = tape.gradient(loss_value_disc, self.disc_model.trainable_variables)

        self.disc_optimizer.apply_gradients(
            zip(
                grads_disc,
                self.disc_model.trainable_variables,
            ),
            global_step=tf.train.get_or_create_global_step()
        )

        loss = (1 / 2) * (loss_value_gen.numpy() + loss_value_disc.numpy())
        self.brain_history.append(loss)
        return loss

    def train_gen(self, imgs, labels):
        """
        trains and returns the loss on the GENERATOR
        :return:
        """

        with tf.GradientTape() as tape:
            predicted = self.gen_model(imgs, training=True)
            loss_value = tf.losses.mean_squared_error(labels, predicted)

        self.gen_loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, self.gen_model.trainable_variables)  # this takes a long time on cpu
        self.gen_optimizer.apply_gradients(
            zip(
                grads,
                self.gen_model.trainable_variables
            ), global_step=tf.train.get_or_create_global_step()
        )

        return loss_value

    def run_training_procedure(self, data_gen):
        # data generator
        batch_count = 0
        # train over time
        # g_epochs = int(self.params.num_epochs * 0.18)
        # d_epochs = int(self.params.num_epochs * 0.02)
        g_epochs = 2
        d_epochs = 2
        for epoch in range(self.params.num_epochs):
            print('\nstarting epoch {}\n'.format(epoch))
            # progress bar visualization (comment out in ML Engine)
            progbar = generic_utils.Progbar(len(data_gen))
            for images, masks, points in data_gen.flow(batch_size=self.params.train_batch_size):

                # batch of images made into a tensor size [batch_size, im_dim_x, im_dim_y, channel)
                images = tf.cast(images, tf.float32)

                # this is the masks in zeros and ones made into a tensor these are [bs, randomx, randomy, 1] shape
                masks = tf.cast(masks, tf.float32)

                # these are the images with the patches blacked out (i.e. set to zero) - same size as images
                erased_imgs = tf.math.multiply(images, tf.math.subtract(tf.constant(1, dtype=tf.float32), masks))

                # we create the "roi_imgs" which are the images times a rectange of size self.local_shape which
                # encompasses the entire mask. Note all rectangles are the same size. we use "points" defined in the
                # generator to create such images. then cast it to a tensor
                # size [bs, local_shape[0], local_shape[1], channels]
                roi_imgs = tf.cast(
                    [
                        tf.image.crop_to_bounding_box(
                            a,
                            offset_height=b[1],
                            offset_width=b[0],
                            target_height=b[3] - b[1],
                            target_width=b[2] - b[0]
                        )
                        for a, b in zip(images, points)
                    ],
                    tf.float32
                )

                # generate predictions on the erased images
                generated_imgs = self.predict_gen(erased_imgs)

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
                    # get the loss from the batch

                    g_loss = self.train_gen(erased_imgs, images)

                # train discriminator alone for 90k epochs
                # then train disc + gen for another 400k epochs. Total of 500k
                else:

                    # not fixed yet
                    # print('warn not yet implemented disc')
                    # cropped_imgs = tf.image.crop_to_
                    d_loss_real = self.train_disc(images, roi_imgs, valid)
                    d_loss_fake = self.train_disc(generated_imgs, roi_imgs, fake)

                    # # throw in real unedited images with label VALID
                    # d_loss_real = self.mng.disc_brain.train_on_batch([images, points], valid)
                    # # throw in A.I. generated images with label FAKE
                    # d_loss_fake = self.mng.disc_brain.train_on_batch([generated_img, points], fake)
                    # # combine and set the disc loss
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    if epoch >= g_epochs + d_epochs:
                        # train the entire brain
                        g_loss = self.train_full_brain(erased_imgs, images, roi_imgs, valid)
                        # g_loss = self.mng.brain.train_on_batch([images, masks, erased_imgs, points], [images, valid])

                # progress bar visualization (comment out in ML Engine)
                # NOTE - d_loss is not .numpy() fied yet - WARN !!!!!!!!!!!!!!!
                progbar.add(int(images.shape[0]), values=[("Disc Loss: ", d_loss), ("Gen Loss: ", g_loss)])
                batch_count += 1

            # gen_brain.save(f"./outputs/models/batch_{batch_count}_generator.h5")
            # disc_brain.save(f"./outputs/models/batch_{batch_count}discriminator.h5")
            if epoch % self.params.epoch_save_frequency == 0 and epoch > 0:
                # save the generated image
                last_img = generated_imgs[0]
                last_img *= 255
                dreamt_image = Image.fromarray(np.asarray(last_img, dtype='uint8'), 'RGB')
                if dreamt_image is not None:
                    output_image_path = 'gs://{}/{}/images/epoch_{}_image.png'.format(
                        self.params.staging_bucketname,
                        self.params.job_dir, epoch
                    )
                    save_img(save_path=output_image_path, img_data=dreamt_image)
