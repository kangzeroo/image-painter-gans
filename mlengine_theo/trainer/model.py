import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Subtract, Add, Multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
import tensorflow.contrib.eager as tfe
from tensorflow.keras.models import Model
import tensorflow.train as optimizers
import tensorflow.keras.backend as K  # there is a odd warning here - BEWARNED
import pdb  # for debugging


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
    disc_model = model_discriminator(global_image_tensor, local_image_tensor, clip_coords)

    disc_model = disc_model([full_img, clip_coords])
    # disc_model

    disc_brain = Model(inputs=[full_img, clip_coords], outputs=disc_model)

    if params.verbosity == 'INFO':
        print(disc_model)
        disc_brain.summary()
        # view_models(disc_brain, '../summaries/disc_brain.png')

    return disc_brain, disc_model


def full_gen_layer(params, full_img, mask, erased_image, global_image_tensor):
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
    gen_model = model_generator(global_image_tensor=global_image_tensor)

    # pass in the erased_image as input
    gen_model = gen_model(erased_image)

    gen_brain = Model(inputs=[full_img, mask, erased_image], outputs=gen_model)

    if params.verbosity == 'INFO':
        print(gen_model)
        print(gen_model)
        print(gen_brain)
        # view_models(gen_brain, '../summaries/gen_brain.png')
        gen_brain.summary()

    return gen_brain, gen_model


def model_generator(global_image_tensor):
    """
    creates the generator
    :param input_shape:
    :return:
    """
    # in_layer = tfe.Variable(global_image_tensor)

    pdb.set_trace()
    model = Conv2D(64, kernel_size=5, strides=1, padding='same',
                     dilation_rate=(1, 1))(global_image_tensor)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(128, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(256, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(4, 4))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(8, 8))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(16, 16))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(128, kernel_size=4, strides=2,
                              padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(64, kernel_size=4, strides=2,
                              padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(32, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(3, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)
    model_gen = Model(inputs=in_layer, outputs=model, name='Gener8tor')
    return model_gen


def model_discriminator(global_shape=(256, 256, 3), local_image_tensor=(128, 128, 3), clip_coords=(1,1,1,1)):
    def crop_image(img, crop):
        # pdb.set_trace()
        return tf.image.crop_to_bounding_box(img,
                                             crop.shape[1],
                                             crop.shape[0],
                                             crop.shape[3] - crop.shape[1],
                                             crop.shape[2] - crop.shape[0])

    in_pts = tfe.Variable((4,), dtype='int32')
    cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                      output_shape=local_image_tensor)
    g_img = tfe.Variable(global_shape)
    # pdb.set_trace()
    # l_img = cropping([g_img, in_pts])
    l_img = tf.image.crop_to_bounding_box(g_img,
                                         clip_coords[1],
                                         clip_coords[0],
                                         clip_coords[3] - clip_coords[1],
                                         clip_coords[2] - clip_coords[0])


    # Local Discriminator
    x_l = Conv2D(64, kernel_size=5, strides=2, padding='same')(l_img)
    # pdb.set_trace()
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    # pdb.set_trace()
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    # pdb.set_trace()
    x_l = Flatten()(x_l)
    # pdb.set_trace()
    # x_l = Reshape()(x_l)
    x_l = Dense(1024, activation='relu')(x_l)

    # Global Discriminator
    x_g = Conv2D(64, kernel_size=5, strides=2, padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    # pdb.set_trace()
    x_g = Flatten()(x_g)
    # pdb.set_trace()
    x_g = Dense(1024, activation='relu')(x_g)

    x = Concatenate(axis=1)([x_l, x_g])
    x = Dense(1, activation='sigmoid')(x)
    model_disc = Model(inputs=[g_img, in_pts], outputs=x, name='Discimi-hater')
    return model_disc


class ModelManager:
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
        self.params = params
        full_img = tfe.Variable(global_image_tensor)
        erased_img = tfe.Variable(global_image_tensor)
        mask = tfe.Variable((global_image_tensor[0], global_image_tensor[1], 1))
        clip_coords = tfe.Variable((4,), dtype='int32')

        self.gen_brain, self.gen_model = full_gen_layer(
            params=params,
            full_img=full_img,
            mask=mask,
            erased_image=erased_img,
            global_image_tensor=global_image_tensor
        )

        self.disc_brain, self.disc_model = full_disc_layer(
            params=params,
            global_image_tensor=global_image_tensor,
            local_image_tensor=local_image_tensor,
            full_img=full_img,
            clip_coords=clip_coords)

        # the final brain
        self.disc_model.trainable = False
        self.connected_disc = Model(inputs=[full_img, clip_coords], outputs=self.disc_model, name='Connected-Discrimi-Hater')

        self.brain = Model(
            inputs=[full_img, mask, erased_img, clip_coords],
            outputs=[
                self.gen_model,
                self.connected_disc([self.gen_model, clip_coords])
            ]
        )
        # we might as well compile here - so just by initializing the class the GAN is completely ready to go
        self.compile()

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
