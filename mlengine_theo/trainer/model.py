import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Subtract, Add, Multiply
from tensorflow.keras.layers import BatchNormalization
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import Model
import tensorflow.train as optimizers
import tensorflow.keras.backend as K  # there is a odd warning here - BEWARNED
from keras.utils import generic_utils
import pdb  # for debugging
import numpy as np


def full_disc_layer(params, global_image_tensor, local_image_tensor, full_img, clip_coords):
    """
    creates the discriminator
    :param params: DICT - from argparser contains most paramaters
    :param global_shape: tuple - assumed RGB
    :param local_image_tensor: tuple - assumed RGB
    :param full_img:
    :param clip_coords:
    :return:
    """
    x_l, x_g = model_discriminator()
    # x = Concatenate(axis=1)([x_l, x_g])
    # x = Dense(1, activation='sigmoid')(x)

    # model_disc = Model(inputs=[g_img, in_pts], outputs=x, name='Discimi-hater')

    # disc_model = disc_model([full_img, clip_coords])
    # disc_model

    # disc_brain = Model(inputs=[full_img, clip_coords], outputs=disc_model)

    # if params.verbosity == 'INFO':
    #     print(disc_model)
    #     disc_brain.summary()
    #     # view_models(disc_brain, '../summaries/disc_brain.png')

    return x_l, x_g


def full_gen_layer():
    """
    oversees the generator creation
    :param params:
    :param full_img:
    :param mask:
    :param erased_image:
    :param global_shape:
    :return:
    """
    # view our net
    gen_model = model_generator()

    # model_gen = Model(inputs=in_layer, outputs=gen_model, name='Gener8tor')

    # # pass in the erased_image as input
    # gen_model = gen_model(erased_image)

    # gen_brain = Model(inputs=[full_img, mask, erased_image], outputs=gen_model)

    # if params.verbosity == 'INFO':
    #     print(gen_model)
    #     print(gen_model)
    #     print(gen_brain)
    #     # view_models(gen_brain, '../summaries/gen_brain.png')
    #     gen_brain.summary()

    return gen_model


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

    # x = Concatenate(axis=1)([x_l, x_g])
    # x = Dense(1, activation='sigmoid')(x)

    # model_disc = Model(inputs=[g_img, in_pts], outputs=x, name='Discimi-hater')
    return x_l, x_g


class ModelManager(Model):
    """
    this class basically just holds the GAN ... later we can build out more functionality like train, predict etc...
    """
    def __init__(
        self,
        params,
        global_image_tensor,
        local_image_tensor,
    ):
        """
        manage the gan. we construct the GAN on initialization.......
        :param params: DICT - from argparser
        """
        super(ModelManager, self).__init__()
        self.params = params
        full_img = tfe.Variable(global_image_tensor)
        erased_img = tfe.Variable(global_image_tensor)
        mask = tfe.Variable((global_image_tensor[0], global_image_tensor[1], 1))
        clip_coords = tfe.Variable((4,), dtype='int32')

        self.gen_model = full_gen_layer()

        self.x_l, self.x_g = full_disc_layer(
            params=params,
            global_image_tensor=global_image_tensor,
            local_image_tensor=local_image_tensor,
            full_img=full_img,
            clip_coords=clip_coords
        )
        # x = Concatenate(axis=1)([x_l, x_g])
        # x = Dense(1, activation='sigmoid')(x)

        # model_disc = Model(inputs=[g_img, in_pts], outputs=x, name='Discimi-hater')

        # disc_model = disc_model([full_img, clip_coords])
        # disc_model

        # disc_brain = Model(inputs=[full_img, clip_coords], outputs=disc_model)

        # if params.verbosity == 'INFO':
        #     print(disc_model)
        #     disc_brain.summary()
        #     # view_models(disc_brain, '../summaries/disc_brain.png')

    def predict(self, img):
        """
        predict the generator on a given image
        :param erased_imgs:
        :return:
        """

        generated_img = self.gen_model(img)
        return generated_img

    def train(self, data_gen):
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
            progbar = generic_utils.Progbar(len(data_gen))
            for images, points, masks in data_gen.flow(batch_size=self.params.train_batch_size):
                masks_inv = [1 - mask for mask in masks]
                erased_imgs = np.asarray([img * mask_inv for img, mask_inv in zip(images, masks)])
                # generate the inputs (images)
                pdb.set_trace()
                generated_img = self.predict(erased_imgs)
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

    # def call(self, input, points, mask):
    #     """
    #     run / connect the model on the fly
    #     assume just one image, mask etc now...
    #     :param input: batch of inputs????? right now assumed one image
    #     :return:
    #     """
    #     # we basically connect everything here on the fly it would seem?
    #     # but first we want a predicted image
    #     #
    #     erased_img = input * mask
    #     generated_img = self.predict(erased_img)
    #
    #
    #     # run the original image through
    #     # first we need to get the loss of the disc
    #     # but we need to create it first:
    #     result_xl = self.x_l(input)
    #     result_xg = self.x_g(input)
    #     # concatenate it
    #     result_disc = Concatenate(axis=1)([result_xl, result_xg])
    #     result_orig = Dense(1, activation='sigmoid')(result_disc)
    #
    #     # now maybe the mask
    #
    #
    #
    #     # OKKKKKKAY
    #     # disc_brain = Model(inputs=[full_img, clip_coords], outputs=disc_model)
    #     # now get the losss
    #
    #
    #
    #
    #
    #
    #         # throw in real unedited images with label VALID
    #         d_loss_real = self.mng.disc_brain.train_on_batch([images, points], valid)
    #         # throw in A.I. generated images with label FAKE
    #         d_loss_fake = self.mng.disc_brain.train_on_batch([generated_img, points], fake)
    #         # combine and set the disc loss
    #         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    #         if epoch >= g_epochs + d_epochs:
    #             # train the entire brain
    #             g_loss = self.mng.brain.train_on_batch([images, masks, erased_imgs, points], [images, valid])
    #             # and update the generator loss
    #             g_loss = g_loss[0] + self.params.alpha * g_loss[1]
    #     # progress bar visualization (comment out in ML Engine)
    #     progbar.add(images.shape[0], values=[("Disc Loss: ", d_loss), ("Gen mse: ", g_loss)])
    #     batch_count += 1
    #     # save the generated image
    #     last_img = generated_img[0]
    #     last_img *= 255
    #     dreamt_image = Image.fromarray(last_img.astype('uint8'), 'RGB')
    #
    #
    #
    #
    #
    #     result = self.dense1(input)
    #     result = self.dense2(result)
    #     result = self.dense2(result)  # reuse variables from dense2 layer
    #     return result
    #
    #
    #     # old
    #     # self.disc_brain, self.disc_model = full_disc_layer(
    #     #     params=params,
    #     #     global_image_tensor=global_image_tensor,
    #     #     local_image_tensor=local_image_tensor,
    #     #     full_img=full_img,
    #     #     clip_coords=clip_coords)
    #
    #     # the final brain
    #     self.disc_model.trainable = False
    #     self.connected_disc = Model(inputs=[full_img, clip_coords], outputs=self.disc_model, name='Connected-Discrimi-Hater')
    #
    #     self.brain = Model(
    #         inputs=[full_img, mask, erased_img, clip_coords],
    #         outputs=[
    #             self.gen_model,
    #             self.connected_disc([self.gen_model, clip_coords])
    #         ]
    #     )
    #     # we might as well compile here - so just by initializing the class the GAN is completely ready to go
    #     self.compile()

    def compile(self):
        """
        compiles the generator brain and the brain itself ... ?

        ..later we can make it compile each one seperately if we want
        :return:
        """
        print('compilling hater')
        self.disc_brain.compile(
            loss=self.params.disc_loss,
            optimizer=getattr(optimizers, self.params.optimizer)(learning_rate=self.params.learning_rate)
        )
        # optimizer=optimizers.Adadelta(lr=0.01))
        print('compilling generator')
        self.gen_brain.compile(
            loss=self.params.gen_loss,
            optimizer=getattr(optimizers, self.params.optimizer)(learning_rate=self.params.learning_rate)
            # optimizer=optimizers.Adadelta(lr=0.001)
        )
        print('compilling brain ---- note some hardcodded loss up in here')
        self.brain.compile(
            loss=['mse', 'binary_crossentropy'],
            loss_weights=[1.0, self.params.alpha],
            optimizer=getattr(optimizers, self.params.optimizer)(learning_rate=self.params.learning_rate)
        )

    def describe(self):
        """
        describe your gan ?
        :return:
        """
        if self.params.verbosity == 'INFO':
            print(self.gen_brain)
            print(self.disc_brain)

            print(self.gen_model)
            print(self.disc_model)
            print(self.connected_disc)
            self.brain.summary()
            # view_models(brain, '../summaries/brain.png')

    # def train(self):
