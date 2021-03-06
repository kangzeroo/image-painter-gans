import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.layers import Flatten, Activation, Conv2D, Conv2DTranspose, Dense, BatchNormalization
from tensorflow.keras import Model


try:
    from utils import save_img, extract_roi_imgs
except Exception as e:
    from trainer.utils import save_img, extract_roi_imgs


def log_scalar(name, val, logging_frequency=1):
    """
    tensorboard logs "name" with value = val
    :param name: str - name of paramater
    :param val: value of paramater (scalar i.e. loss)
    :return:
    """
    # ummm... does this work?
    _ = 1
    with tf.contrib.summary.record_summaries_every_n_global_steps(logging_frequency):
        tf.contrib.summary.scalar(name, val)


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
        training = tfe.Variable(training)  # BOOL tensor - need this to be TRUE for the tf.keras.layers.BatchNormalization() to work at all .... seems bugged

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
        optimizer='AdadeltaOptimizer',
        lr=1.0,
        alpha=0.0004,
        load_ckpt_dir=None,
        tb_log_dir='./tb_logs/'
    ):
        """
        manage the gan. we construct the discriminator and generator on initialization....... Note, these are not "compiled"
        yet, since we are using eager, they only get compiled when called
        :param optimizer: str - specifying module in tf.train.optimizer
            * NOTE - as of now, this is used for BOTH GENERATOR and DISCRIMINATOR (which i think is how its in paper anyways??? )
        :param lr: float - learning rate for optimizer
            * NOTE - used for both gen and disc (i think as per paper also?)
        :param alpha: float - weighting on discriminator loss it would seem
        :param load_ckpt_dir: str or None - if not None, path to ckpt to be picked up.
        :param tb_log_dir: str or None - if not None, logs the losses etc in the directory specified.
        """
        # super(ModelManager, self).__init__()
        self.alpha = alpha
        self.gen_loss_history, self.disc_loss_history, self.brain_history = [], [], []
        # optimizers ***** NOTE THESE might be different from paper
        # NOTE paper: if using tf.losses.AdadeltaOptimizer use a learning rate of 1.0
        self.gen_optimizer = getattr(tf.train, optimizer)(learning_rate=lr)
        self.gen_optimizer.__setattr__('name', 'gen_optimizer')  # necessary for checkpoint???
        self.disc_optimizer = getattr(tf.train, optimizer)(learning_rate=lr)
        self.disc_optimizer.__setattr__('name', 'disc_optimizer')
        # this is the generator model
        self.gen_model = Generator()  # need to checkpoint this... NOTE --- it's name is "generator_model"
        # full discriminator (i.e. global + local branch)
        self.disc_model = DiscConnected()  # need to checkpoint this... NOTE --- its name is disc_connected

        # keep track of epoch ----- overides in task.py run_job()
        self.epoch = tfe.Variable(0, name='epoch', dtype=tf.int64)  # if loading in the checkpoint, we will set self.epoch with the save epoch value

        # first sort out tensorboard logging
        self.tb_logger = tf.contrib.summary.create_file_writer(tb_log_dir)
        self.tb_logger.set_as_default()


        # then checkpointing
        # we will create keyword args to throw into the checkpoint using the names of the self variables above
        # the keys are the names above and the values are the self.$varname

        kwarg = {
            'gen_optimizer': self.gen_optimizer,
            'disc_optimizer': self.disc_optimizer,
            'generator_model': self.gen_model,
            'disc_connected': self.disc_model,
            'epoch': self.epoch
        }
        self.checkpoint = tf.train.Checkpoint(**kwarg)

        # now if param use checkpoint is true, load up the checkpoint
        # in theory, this will alter all of the state variables defined above!
        if load_ckpt_dir is not None:
            print('RESTORING FROM CHECKPOINT from {}'.format(load_ckpt_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(load_ckpt_dir))
            print('sanity check - loaded epoch ... {}'.format(self.epoch))

        else:
            print('WARN ---- not picking up any checkpoints')
            # we should really delete the folder contents in this case ...

    def _gen_loss(self, x, y, training=True):
        """
        calculate the generator loss
        :param x: tensor - masked image in batches (i.e. with roi set to zero)
        :param y: tensor - the acutal images
        :param training: bool - to train or not to? (NOTE always set this to true, otherwise
            inferrence will yield nans (potential bug))) BITCH
        :return: mse loss, predictions
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
        :return: sigmoid xentropy loss
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

        log_scalar('generator_loss', loss_value.numpy())
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

        log_scalar('discriminator_loss', loss_value.numpy())
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
                loss = tf.add(loss_value_gen, tf.multiply(loss_value_disc, self.alpha))

            log_scalar('combined_loss', loss.numpy())
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
