import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.layers import Flatten, Activation, Conv2D, Conv2DTranspose, Dense, BatchNormalization
from tensorflow.keras.models import Model
from keras.utils import generic_utils
import pdb  # for debugging
import numpy as np
from PIL import Image
from os.path import join as jp

try:
    from utils import save_img, extract_roi_imgs
except Exception as e:
    from trainer.utils import save_img, extract_roi_imgs


# hardcoded..... lets put this in a config along with other shit later on
print('warn hardcoded ckpt folder')
ckpt_fol = 'ckpt'


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
        Activation('relu'),
        Conv2D(3, kernel_size=3, strides=1,
               padding='same', dilation_rate=(1, 1)),
        BatchNormalization(),
        Activation('sigmoid')
    ], name='generator_model')

    # model_gen = Model(inputs=in_layer, outputs=model, name='Gener8tor')
    return model_gen


def model_discriminator():

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

#
# class GLCIC(Model):
#     def __init__(self, gen_model, disc_model):
#         super(GLCIC, self).__init__()
#         self.gen_model
#

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen_model = model_generator()

    def call(self, inputs, training=False):
        x = self.gen_model(inputs)
        return x


class DiscConnected(Model):
    """
    the connected discriminator
        - local discriminator branch concatenated with
        - global discriminator  branch

    """
    def __init__(self):
        super(DiscConnected, self).__init__()
        self.disc_model_local, self.disc_model_global = model_discriminator()

    def call(self, inputs, cropped_imgs, training=False):
        # x = tf.keras.cont(inputs)
        x = tf.concat([self.disc_model_local(cropped_imgs), self.disc_model_global(inputs)], axis=1)
        x = Dense(1, activation='sigmoid')(x)
        return x


class ModelManager:
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
        # super(ModelManager, self).__init__()
        self.params = params
        load_ckpt = self.params.load_ckpt  # this is inputted and tells if to load or not to load
        self.gen_loss_history, self.disc_loss_history, self.brain_history = [], [], []

        # a note on base_save_dir -> normally, we do not need the "//" after "gs://" because os automatically infers it...
        # however, using google storage, it throws an error without the front slashes.... so keep there there
        self.base_save_dir = jp('gs://', self.params.staging_bucketname, self.params.job_dir)  # use this to construct paths if needed
        self.ckpt_dir = jp(self.base_save_dir, 'ckpt/')  # where we throw out checkpoints AND SAME NOTE as above ~~~

        # optimizers ***** NOTE THESE might be different from paper
        self.gen_optimizer = getattr(tf.train, self.params.optimizer)(learning_rate=self.params.learning_rate)
        self.gen_optimizer.__setattr__('name', 'gen_optimizer')  # necessary for checkpoint???
        self.disc_optimizer = getattr(tf.train, self.params.optimizer)(learning_rate=self.params.learning_rate)  # NOTE THIS might be different from paper
        self.disc_optimizer.__setattr__('name', 'disc_optimizer')
        # this is the generator model
        self.gen_model = Generator()  # need to checkpoint this... NOTE --- it's name is "generator_model"
        # full discriminator (i.e. global + local branch)
        self.disc_model = DiscConnected()  # need to checkpoint this... NOTE --- its name is disc_connected

        # we need to explicitly global steps for each model
        # self.disc_gs = tfe.Variable(0, name='disc_gs', dtype=tf.int64)  # we can save this in checkpoint
        # self.gen_gs = tfe.Variable(0, name='gen_gs', dtype=tf.int64)  # we can save this in checkpoint
        # self.global_step = tfe.Variable(0, name='gen_gs', dtype=tf.int64)
        # keep track of "GLOBAL" epoch ----- overides in run_training_procedure
        self.epoch = tfe.Variable(0, name='overall_epoch', dtype=tf.int64)  # if loading in the checkpoint, we will set self.epoch with the save epoch value

        # we will create keyword args to throw into the checkpoint using the names of the self variables above
        # the keys are the names above and the values are the self.$varname

        kwarg = {
            'gen_optimizer': self.gen_optimizer,
            'disc_optimizer': self.disc_optimizer,
            'generator_model': self.gen_model,
            'disc_connected': self.disc_model,
            'overall_epoch': self.epoch
        }

        self.checkpoint = tf.train.Checkpoint(**kwarg)

        # now if param use checkpoint is true, load up the checkpoint
        # in theory, this will alter all of the state variables defined above!
        if load_ckpt:
            print('RESTORING FROM CHECKPOINT from {}'.format(self.ckpt_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_dir))
            print('sanity check - loaded epoch ... {}'.format(self.epoch))
        else:
            # DELETE THE DIRECTORY ....
            print('WARN ---- not picking up any checkpoints')
            # we should really delete the folder contents in this case ...

    def train_gen(self, imgs, labels):
        """
        trains and returns the loss on the GENERATOR
        :return:
        """

        with tf.GradientTape() as tape:
            predicted = self.gen_model(imgs, training=True)
            loss_value = tf.losses.mean_squared_error(labels, predicted)

        self.gen_loss_history.append(loss_value.numpy())  # track the loss in a variable
        grads = tape.gradient(loss_value, self.gen_model.trainable_variables)  # this takes a long time on cpu
        self.gen_optimizer.apply_gradients(
            zip(
                grads,
                self.gen_model.trainable_variables
            ),
            global_step=tf.train.get_or_create_global_step()
        )
        return loss_value

    def train_disc(self, imgs, masked_imgs, labels):
        """
        we kinda need to create it on the fly also
        :param imgs:
        :param masked_imgs:
        :param labels:
        :return:
        """

        with tf.GradientTape() as tape:
            output = self.disc_model(imgs, masked_imgs)
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

    def train_full_brain(self, erased_imgs, images, points, valid):
        """
        trains the entire brain (i.e. gen + disc). can make probably write this more general and combine the other 2 training methods....

        WARN - this might not be fully correct.... might need to do "joint" loss etc

        :param erased_imgs:
        :param images:
        :param roi_imgs:
        :param valid:
        :return:
        """
        with tf.GradientTape() as tape_gen:
            # output_gen, output_disc = self.full_brain(erased_imgs, roi_imgs)
            tape_gen.watch(self.gen_model.variables)
            output_gen = self.gen_model(erased_imgs)
            loss_value_gen = tf.losses.mean_squared_error(
                images,
                output_gen,
                # weights=1.0
            )

            with tf.GradientTape() as tape_disc:

                tape_disc.watch(self.disc_model.variables)
                # get the roi of the generator output
                roi_imgs_gen = extract_roi_imgs(output_gen, points)

                output_disc = self.disc_model(output_gen, roi_imgs_gen)
                loss_value_disc = tf.losses.sigmoid_cross_entropy(
                    # we use sigmoid_cross_entropy in replace of keras' binary cross entropy
                    valid,
                    output_disc,
                    # weights=self.params.alpha
                )

                loss = tf.add(loss_value_gen, tf.multiply(loss_value_disc, self.params.alpha))
                # loss = tf.add(loss_value_gen, loss_value_disc)

            # I think we really just train the generator
            # train the generator
            grads_gen = tape_gen.gradient(loss, self.gen_model.trainable_variables)
            self.gen_optimizer.apply_gradients(
                zip(
                    grads_gen,
                    self.gen_model.trainable_variables,
                ),
                global_step=tf.train.get_or_create_global_step()
            )

        # # now lets teach the discriminator...
        # grads_disc = tape_disc.gradient(loss, self.disc_model.trainable_variables)
        # self.disc_optimizer.apply_gradients(
        #     zip(
        #         grads_disc,
        #         self.disc_model.trainable_variables,
        #     ),
        #     global_step=tf.train.get_or_create_global_step()
        # )
        # # is this right loss?

        self.brain_history.append(loss)
        return loss, loss_value_gen

    def run_training_procedure(self, data_gen):
        # train over time
        # g_epochs = int(self.params.num_epochs * 0.18)
        # d_epochs = int(self.params.num_epochs * 0.02)
        g_epochs = 2
        d_epochs = 2
        generated_imgs = None  # redundant... but to stop warning
        init_epoch = self.epoch.numpy()
        for epoch in range(init_epoch, self.params.num_epochs):
            print('\nstarting epoch {}\n'.format(epoch))
            # progress bar visualization (comment out in ML Engine)
            prog_cap = 300000000 if self.params.max_img_cnt is None else self.params.max_img_cnt
            progbar = generic_utils.Progbar(prog_cap)
            for images, masks, points in data_gen.flow(batch_size=self.params.train_batch_size):

                # batch of images made into a tensor size [batch_size, im_dim_x, im_dim_y, channel)
                images = tf.cast(images, tf.float32)

                # this is the masks in zeros and ones made into a tensor these are [bs, randomx, randomy, 1] shape
                masks = tf.cast(masks, tf.float32)

                # these are the images with the patches blacked out (i.e. set to zero) - same size as images
                erased_imgs = tf.multiply(images, tf.subtract(tf.constant(1, dtype=tf.float32), masks))

                # we create the "roi_imgs" which are the images times a rectange of size self.local_shape which
                # encompasses the entire mask. Note all rectangles are the same size. we use "points" defined in the
                # generator to create such images. then cast it to a tensor
                # size [bs, local_shape[0], local_shape[1], channels]

                # generate predictions on the erased images
                generated_imgs = self.gen_model(erased_imgs)

                # generate the labels
                valid = np.ones((self.params.train_batch_size, 1))
                fake = np.zeros((self.params.train_batch_size, 1))
                # the gen and disc losses
                g_loss = tfe.Variable(0)
                d_loss = tfe.Variable(0)

                # we must train the neural nets seperately, and then together
                # train generator for 90k epochs
                if epoch < g_epochs:
                    # set the gen loss
                    # get the loss from the batch

                    g_loss = self.train_gen(erased_imgs, images)

                # train discriminator alone for 90k epochs
                # then train disc + gen for another 400k epochs. Total of 500k
                else:
                    roi_imgs_real, roi_imgs_fake = extract_roi_imgs(images, points), extract_roi_imgs(erased_imgs, points)
                    d_loss_real = self.train_disc(images, roi_imgs_real, valid)
                    d_loss_fake = self.train_disc(generated_imgs, roi_imgs_fake, fake)

                    # # throw in real unedited images with label VALID
                    # d_loss_real = self.mng.disc_brain.train_on_batch([images, points], valid)
                    # # throw in A.I. generated images with label FAKE
                    # d_loss_fake = self.mng.disc_brain.train_on_batch([generated_img, points], fake)
                    # # combine and set the disc loss
                    d_loss = tf.multiply(tf.add(d_loss_real, d_loss_fake), 0.5)
                    if epoch >= g_epochs + d_epochs:
                        # train the entire brain
                        combined_loss, g_loss = self.train_full_brain(erased_imgs, images, points, fake)
                        # g_loss = self.mng.brain.train_on_batch([images, masks, erased_imgs, points], [images, valid])

                # progress bar visualization (comment out in ML Engine)
                progbar.add(int(images.shape[0]), values=[("Disc Loss: ", d_loss.numpy()), ("Gen Loss: ", g_loss.numpy()), ("Combined Loss: ", combined_loss.numpy())])

            # increment the self.epoch  -> we need to do this so that the checkpoint is accurate....
            self.epoch.assign_add(1)
            if epoch % self.params.epoch_save_frequency == 0 and epoch > 0:
                # save check_point
                print('saving checkpoint {}'.format(self.ckpt_dir))
                self.checkpoint.save(self.ckpt_dir)
                # save a generated image for peace of mind
                if generated_imgs is not None:
                    last_img = generated_imgs[0]
                    last_img *= 255
                    dreamt_image = Image.fromarray(np.asarray(last_img, dtype='uint8'), 'RGB')
                    output_image_path = jp(self.base_save_dir, 'images', 'epoch_{}_image.png'.format(epoch))
                    save_img(save_path=output_image_path, img_data=dreamt_image)
