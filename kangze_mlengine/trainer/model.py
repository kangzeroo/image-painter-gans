import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.layers import Flatten, Activation, Conv2D, Conv2DTranspose, Dense, BatchNormalization
from tensorflow.keras import Model
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


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()

        # self.name = 'generator_model'  # how do you set this

        self.conv11 = Conv2D(64, kernel_size=5, strides=1, padding='same',
                            dilation_rate=(1, 1))
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')

        self.conv21 = Conv2D(128, kernel_size=3, strides=2,
                             padding='same', dilation_rate=(1, 1))
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

        self.conv22 = Conv2D(128, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')

        self.conv31 = Conv2D(256, kernel_size=3, strides=2,
                             padding='same', dilation_rate=(1, 1))
        self.bn4 = BatchNormalization()
        self.act4 = Activation('relu')

        self.conv32 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn5 = BatchNormalization()
        self.act5 = Activation('relu')

        self.conv33 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn6 = BatchNormalization()
        self.act6 = Activation('relu')

        self.conv34 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(2, 2))
        self.bn7 = BatchNormalization()
        self.act7 = Activation('relu')

        self.conv35 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(4, 4))
        self.bn8 = BatchNormalization()
        self.act8 = Activation('relu')

        self.conv36 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(8, 8))
        self.bn9 = BatchNormalization()
        self.act9 = Activation('relu')

        self.conv37 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(16, 16))
        self.bn10 = BatchNormalization()
        self.act10 = Activation('relu')

        self.conv38 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn11 = BatchNormalization()
        self.act11 = Activation('relu')

        self.conv39 = Conv2D(256, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn12 = BatchNormalization()
        self.act12 = Activation('relu')

        self.convT1 = Conv2DTranspose(128, kernel_size=4, strides=2,
                                      padding='same')
        self.bn13 = BatchNormalization()
        self.act13 = Activation('relu')

        self.conv41 = Conv2D(128, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn14 = BatchNormalization()
        self.act14 = Activation('relu')

        self.convT2 = Conv2DTranspose(64, kernel_size=4, strides=2,
                                   padding='same')
        self.bn15 = BatchNormalization()
        self.act15 = Activation('relu')

        self.conv51 = Conv2D(32, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn16 = BatchNormalization()
        self.act16 = Activation('relu')

        self.conv61 = Conv2D(3, kernel_size=3, strides=1,
                             padding='same', dilation_rate=(1, 1))
        self.bn17 = BatchNormalization()
        self.act17 = Activation('sigmoid')

    def call(self, inputs, training=True, *args, **kwargs):
        """
        ummm..... CALL THIS BITCH .... note - training should always be true, otherwise, batchnorm layers will
        yields nans after consequitive calls (BUG???)
        :param inputs: tensor - should be the MASKED imgs in batchs
        :param training: BOOL - set to true in order for the bathcnorm layers to work...
        :param args:
        :param kwargs:
        :return:
        """
        training = tfe.Variable(training)

        x = self.conv11(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv21(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv22(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)

        x = self.conv31(x)
        x = self.bn4(x, training=training)
        x = self.act4(x)

        x = self.conv32(x)
        x = self.bn5(x, training=training)
        x = self.act5(x)

        x = self.conv33(x)
        x = self.bn6(x, training=training)
        x = self.act6(x)

        x = self.conv34(x)
        x = self.bn7(x, training=training)
        x = self.act7(x)

        x = self.conv35(x)
        x = self.bn8(x, training=training)
        x = self.act8(x)

        x = self.conv36(x)
        x = self.bn9(x, training=training)
        x = self.act9(x)

        x = self.conv37(x)
        x = self.bn10(x, training=training)
        x = self.act10(x)

        x = self.conv38(x)
        x = self.bn11(x, training=training)
        x = self.act11(x)

        x = self.conv39(x)
        x = self.bn12(x, training=training)
        x = self.act12(x)

        x = self.convT1(x)
        x = self.bn13(x, training=training)
        x = self.act13(x)

        x = self.conv41(x)
        x = self.bn14(x, training=training)
        x = self.act14(x)

        x = self.convT2(x)
        x = self.bn15(x, training=training)
        x = self.act15(x)

        x = self.conv51(x)
        x = self.bn16(x, training=training)
        x = self.act16(x)

        x = self.conv61(x)
        x = self.bn17(x, training=training)
        x = self.act17(x)

        return x


class DiscConnected(Model):
    """
    the connected discriminator
        - local discriminator branch concatenated with
        - global discriminator  branch

    """
    def __init__(self):
        super(DiscConnected, self).__init__()
        # self.disc_model_local, self.disc_model_global = model_discriminator()
        # Local Discriminator
        self.lc1 = Conv2D(64, kernel_size=5, strides=2, padding='same')
        self.lbn1 = BatchNormalization()
        self.lact1 = Activation('relu')
        self.lc2 = Conv2D(128, kernel_size=5, strides=2, padding='same')
        self.lbn2 = BatchNormalization()
        self.lact2 = Activation('relu')
        self.lc3 = Conv2D(256, kernel_size=5, strides=2, padding='same')
        self.lbn3 = BatchNormalization()
        self.lact3 = Activation('relu')
        self.lc40 = Conv2D(512, kernel_size=5, strides=2, padding='same')
        self.lbn4 = BatchNormalization()
        self.lact4 = Activation('relu')
        self.lc41 = Conv2D(512, kernel_size=5, strides=2, padding='same')
        self.lbn5 = BatchNormalization()
        self.lact5 = Activation('relu')
        self.lf1 = Flatten()
        self.ld1 = Dense(1024, activation='relu')

        # Global Discriminator
        self.gc1 = Conv2D(64, kernel_size=5, strides=2, padding='same')
        self.gbn1 = BatchNormalization()
        self.gact1 = x_l = Activation('relu')
        self.gc2 = Conv2D(128, kernel_size=5, strides=2, padding='same')
        self.gbn2 = BatchNormalization()
        self.gact2 = Activation('relu')
        self.gc3 = Conv2D(256, kernel_size=5, strides=2, padding='same')
        self.gbn3 = BatchNormalization()
        self.gact3 = Activation('relu')
        self.gc40 = Conv2D(512, kernel_size=5, strides=2, padding='same')
        self.gbn4 = BatchNormalization()
        self.gact4 = Activation('relu')
        self.gc41 = Conv2D(512, kernel_size=5, strides=2, padding='same')
        self.gbn5 = BatchNormalization()
        self.gact5 = Activation('relu')
        self.gc42 = Conv2D(512, kernel_size=5, strides=2, padding='same')
        self.gbn6 = BatchNormalization()
        self.gact6 = Activation('relu')
        self.gf1 = Flatten()
        self.gd1 = Dense(1024, activation='relu')

    def call(self, inputs, cropped_imgs, training=True,  *args, **kwargs):
        """
        call this bitch, bitch. NOTE - batch norm needs paraing "training" to be True... otherwise, yields nans
        after consequetive calls (potential bug?)
        :param inputs: tensor - imgs from gnerator (presumadly cunt) ni batches
        :param cropped_imgs: tensor - imgs in batch the cropped images bitch
        :param training: BOOL - for the fucking batchnorms cunt
        :param args:
        :param kwargs:
        :return:
        """
        training = tfe.Variable(training)

        # local
        xl = self.lc1(cropped_imgs)
        xl = self.lbn1(xl, training=training)
        xl = self.lact1(xl)
        xl = self.lc2(xl)
        xl = self.lbn2(xl, training=training)
        xl = self.lact2(xl)
        xl = self.lc3(xl)
        xl = self.lbn3(xl, training=training)
        xl = self.lact3(xl)
        xl = self.lc40(xl)
        xl = self.lbn4(xl, training=training)
        xl = self.lact4(xl)
        xl = self.lc41(xl)
        xl = self.lbn5(xl, training=training)
        xl = self.lact5(xl)
        xl = self.lf1(xl)
        xl = self.ld1(xl)


        # global
        xg = self.gc1(inputs)
        xg = self.gbn1(xg, training=training)
        xg = self.gact1(xg)
        xg = self.gc2(xg)
        xg = self.gbn2(xg, training=training)
        xg = self.gact2(xg)
        xg = self.gc3(xg)
        xg = self.gbn3(xg, training=training)
        xg = self.gact3(xg)
        xg = self.gc40(xg)
        xg = self.gbn4(xg, training=training)
        xg = self.gact4(xg)
        xg = self.gc41(xg)
        xg = self.gbn5(xg, training=training)
        xg = self.gact5(xg)
        xg = self.gc42(xg)
        xg = self.gbn6(xg, training=training)
        xg = self.gact6(xg)
        xg = self.gf1(xg)
        xg = self.gd1(xg)

        # now lets put these bitches together
        x = tf.concat([xl, xg], axis=1)
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

    def _gen_loss(self, x, y, training=True):
        """
        calculate the generator loss
        :param x: tensor - masked image in batches (i.e. with roi set to zero)
        :param y: tensor - the acutal images
        :param training: bool - to train or not to? (NOTE always set this to true, otherwise
            inferrence will yield nans (potential bug))) BITCH
        :return:
        """
        output = self.gen_model(x, training=training)
        return tf.losses.mean_squared_error(predictions=output, labels=y), output

    def _disc_loss(self, x, xx, y, training=True):
        """
        calculate the loss from the discriminator
        :param x: the global tensor images (in batches)
        :param xx: the local tensor images (batches)
        :param y: the labels (i.e. real or fake BITCH)
        :param training: BOOL - training or not ?? for batchnorm layers - NOTE bugged when this is true....
            PLEASE SET THIS TO TRUE when inffering .... for somereason fucks up when false..... ???? need to look
            into this
        :return:
        """
        return tf.losses.sigmoid_cross_entropy(logits=self.disc_model(x, xx, training=training),
                                               multi_class_labels=y)

    def train_gen(self, imgs, labels):
        """
        trains and returns the loss on the GENERATOR
        :param imgs: tensor - the images with a masked out region
        :param labels: tensor - the actual images (what the generator should reproduce)
        :return:
        """

        with tf.GradientTape() as tape:
            # predicted = self.gen_model(imgs, training=True)
            # loss_value = tf.losses.mean_squared_error(labels, predicted)
            loss_value, _ = self._gen_loss(x=imgs, y=labels, training=True)

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
        TRAIN THE DISCRIMINATORS (global + local).
        :param imgs: tensor - global output from generator (in batches)
        :param masked_imgs: tensor - local output from generator (in batches)
        :param labels: BOOL - whether the imgs + masked_imgs are real or from generator
        :return: LOSS - as tensor from discriminator
        """

        with tf.GradientTape() as tape:
            loss_value = self._disc_loss(imgs, masked_imgs, labels, training=True)

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
        trains the generator on the JOINT (??) loss from the generator and the discriminator.
        :param erased_imgs:
        :param images:
        :param roi_imgs:
        :param valid:
        :return:
        """
        with tf.GradientTape() as tape_gen:
            # first calculate loss + predictions from generator
            loss_value_gen, output_gen = self._gen_loss(erased_imgs, images, training=True)
            # we do a nested gradient tape (???) to track the discriminator also.... prevents errors... ?
            with tf.GradientTape() as tape_disc:
                # get the roi of the generator output
                roi_imgs_gen = extract_roi_imgs(output_gen, points)
                loss_value_disc = self._disc_loss(output_gen, roi_imgs_gen, valid, training=True)
                # combine the losses with the paramater alpha... (* SHOULD THIS BE HERE? or in weights.. ? )
                loss = tf.add(loss_value_gen, tf.multiply(loss_value_disc, self.params.alpha))

            # train the generator (note - discriminator has already been trainined! )
            grads_gen = tape_gen.gradient(loss, self.gen_model.trainable_variables)
            self.gen_optimizer.apply_gradients(
                zip(
                    grads_gen,
                    self.gen_model.trainable_variables,
                ),
                global_step=tf.train.get_or_create_global_step()
            )

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

                # generate predictions on the erased images
                generated_imgs = self.gen_model(erased_imgs, training=True)  # FOR SOME REASON PREDICTING WITH TRAINING=FALSE GIVES NANS

                # generate the labels
                valid = np.ones((self.params.train_batch_size, 1))
                fake = np.zeros((self.params.train_batch_size, 1))
                # the gen and disc losses
                g_loss = tfe.Variable(0)
                d_loss = tfe.Variable(0)
                combined_loss = tfe.Variable(0)

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
