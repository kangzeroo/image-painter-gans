import argparse
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import numpy as np
from datetime import datetime
from keras.utils import generic_utils
from PIL import Image

try:
    from utils import save_img, extract_roi_imgs, save_model, log_scalar
except Exception as e:
    from trainer.utils import save_img, extract_roi_imgs, save_model, log_scalar


# enable eager execution......
# QUESTION - do we need to call this in model.py also for example???
tf.enable_eager_execution()
tf.executing_eagerly()

# NOTE -- for some fucking ass reason, need to change the import names when running local versus cloud............
#         local as in this modules README at least....
#
#         so lets wrap the import in a try catch
try:
    from model import ModelManager
    from generator import DataGenerator

except Exception as e:
    # # cloud - multi
    from trainer.model import ModelManager
    from trainer.generator import DataGenerator


def initialize_hyper_params(args_parser):

    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Args:
        args_parser
    """

    # Data files arguments
    args_parser.add_argument(
        '--job-name',
        help='Current gcloud job name',
        type=str,
    )
    args_parser.add_argument(
        '--train-batch-size',
        help='Batch size for each training step',
        type=int,
        default=1  # currently 25 throws memory errors...... NEED TO INCREASE THIS BABY (use 20 for now)
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
        '--steps-per-epoch',
        help="""\
            Number of steps per epoch.
            """,
        default=3,
        type=int,
    )
    args_parser.add_argument(
        '--gen-loss',
        help="""\
            The loss function for generator\
            * THIS IS NOT hooked up (should be implemented) -- currently hardcoded in model.py
            """,
        default='mse',
        type=str,
    )
    args_parser.add_argument(
        '--disc-loss',
        help="""\
            The loss function for discriminator\
            * THIS IS NOT hooked up (should be implemented) -- currently hardcoded in model.py
            """,
        default='binary_crossentropy',
        type=str,
    )
    args_parser.add_argument(
        '--alpha',
        default=0.0004,  # should not change this
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
        default=10,
        type=int,
    )
    args_parser.add_argument(
        '--job-dir',
        # default="gs://temp/outputs",
        default="testing_baby_newnewnew",
        type=str,
    )
    args_parser.add_argument(
        '--img-dir',
        # default="gs://temp/outputs",
        default="images/bedroom_val",
        type=str,
    )
    args_parser.add_argument(
        '--load-ckpt',
        default=False,
        type=bool,
        help="""\
            True or False specifying if to load the checkpoint
            file. If True, loads the highest epoch found in the
            ckpt folder in the output dir. If False, goes along
            as normal without loading any checkpoint"""
    )
    args_parser.add_argument(
        '--save-weights',
        help="""\
            Whether or not to save the weights
            """,
        default=False,
        type=bool,
    )
    args_parser.add_argument(
        '--optimizer',
        default='AdadeltaOptimizer',
        help="""\
            The optimizer you want to use. Must be the same
            as in keras.optimizers"""
    )
    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help="Learning rate value for the optimizers",
        # default=0.01,
        default=1.0,
        type=float
    )
    # Estimator arguments
    args_parser.add_argument(
        '--max-img-cnt',
        help="Number of maximum images to look at. Set to None if you"
             "want the whole dataset. Primarily used for testing purposes.",
        default=None,  # NOTE 300 imgs in validation set
        type=int
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


def run_job(params, model_mng, data_gen, base_save_dir, strategy, ckpt_dir=None, g_epochs=1, d_epochs=1):
    """
    run the job ... paramaters are assumed to have been preloaded upon initialization in model_mng.

    Supports model_mng checkpointing, training the generator and discriminator both separately and jointly depending on
    the input paramater epochs (currently defaulted)

    :param params: HYPER_PARAMS inputted
    :param model_mng: model_manager, already initialized
    :param data_gen: data generator
    :param g_epochs: INT - # epochs for generator IN PAPER = int(self.params.num_epochs * 0.18)
    :param d_epochs: INT - # epochs for discriminator IN PAPER = int(self.params.num_epochs * 0.02)
    :return:
    """

    data_generator = data_gen.flow_from_directory(batch_size=params.train_batch_size)

    # train baby bitch
    if g_epochs != int(params.num_epochs * 0.18) or int(params.num_epochs * 0.02):
        print('###### WARN - generator or discriminator epochs are not default as paper!!!')

    # where we save weights (NOT FULLY IMPLEMENTED / VALIDATED)
    model_save_dir = os.path.join(base_save_dir, 'models/')

    generated_imgs = None  # redundant... but to stop warning in IDE
    init_epoch = model_mng.epoch.numpy()
    prog_cap = params.steps_per_epoch*params.train_batch_size if params.max_img_cnt is None else params.max_img_cnt
    progbar = generic_utils.Progbar(prog_cap)
    for epoch in range(init_epoch, params.num_epochs):
        # loop over all the steps
        print('\nstarting epoch {}\n'.format(epoch))
        # progress bar visualization (comment out in ML Engine)
        for __ in range(0, params.steps_per_epoch):

            # for each step, we get the data from the generator
            images, masks, points = next(data_generator)

            # batch of images made into a tensor size [batch_size, im_dim_x, im_dim_y, channel)
            images = tf.cast(images, tf.float32)

            # this is the masks in zeros and ones made into a tensor these are [bs, randomx, randomy, 1] shape
            masks = tf.cast(masks, tf.float32)

            # these are the images with the patches blacked out (i.e. set to zero) - same size as images
            erased_imgs = tf.multiply(images, tf.subtract(tf.constant(1, dtype=tf.float32), masks))

            # generate predictions on the erased images
            generated_imgs = model_mng.gen_model(erased_imgs,
                                            training=True)  # FOR SOME REASON PREDICTING WITH TRAINING=FALSE GIVES NANS

            # generate the labels
            valid = np.ones((params.train_batch_size, 1))
            fake = np.zeros((params.train_batch_size, 1))
            # the gen and disc losses
            g_loss = tfe.Variable(0)
            d_loss = tfe.Variable(0)
            combined_loss = tfe.Variable(0)

            # we must train the neural nets seperately, and then together
            # train generator for 90k epochs
            with strategy.scope():
                if epoch < g_epochs:
                    # train generator
                    g_loss = model_mng.train_gen(erased_imgs, images)

                # train discriminator alone for 90k epochs
                # then train disc + gen for another 400k epochs. Total of 500k
                else:
                    roi_imgs_real, roi_imgs_fake = extract_roi_imgs(images, points), extract_roi_imgs(erased_imgs, points)
                    # train the discriminator
                    d_loss_real = model_mng.train_disc(images, roi_imgs_real, valid)
                    d_loss_fake = model_mng.train_disc(generated_imgs, roi_imgs_fake, fake)

                    # # combine and set the disc loss
                    d_loss = tf.multiply(tf.add(d_loss_real, d_loss_fake), 0.5)
                    log_scalar('discriminator_loss', d_loss)
                    if epoch >= g_epochs + d_epochs:
                        # train the entire brain (note this only updates the generator - but uses joint loss gen + disc)
                        combined_loss, g_loss = model_mng.train_full_brain(erased_imgs, images, points, fake)

            # progress bar visualization (comment out in ML Engine)
            progbar.add(int(images.shape[0]), values=[("Disc Loss: ", d_loss.numpy()), ("Gen Loss: ", g_loss.numpy()),
                                                      ("Combined Loss: ", combined_loss.numpy())])

        # increment the self.epoch  -> we need to do this so that the checkpoint is accurate....
        model_mng.epoch.assign_add(1)  # note this might be stupid --- can lead to desynchronization ...
        # might consider just setting model_mng.epoch = tensor(epoch) for example.
        if epoch % params.epoch_save_frequency == 0 and epoch > 0:
            # save check_point - THESE GET PICKED UP IF SPECIFIED
            print('\nsaving checkpoint {}\n'.format(ckpt_dir))
            model_mng.checkpoint.save(ckpt_dir)
            if params.save_weights:
                # NOTE - this is not full implemented or tested ~~~~~~~~
                # save the model weights and architecture... these dont get used because we pick up checkpoints FYI
                # note this is pretty slow tbh
                save_model(
                    model=model_mng.gen_model,
                    save_path=os.path.join(model_save_dir, 'generator_epoch_{}'.format(epoch))
                )
                save_model(
                    model=model_mng.disc_model,
                    save_path=os.path.join(model_save_dir, 'discriminator_epoch_{}'.format(epoch))
                )
            # save a generated image for peace of mind
            if generated_imgs is not None:
                last_img = generated_imgs[0]
                last_img *= 255
                dreamt_image = Image.fromarray(np.asarray(last_img, dtype='uint8'), 'RGB')
                output_image_path = os.path.join(base_save_dir, 'images', 'epoch_{}_image.png'.format(epoch))
                save_img(save_path=output_image_path, img_data=dreamt_image)


def main(params,
         global_shape=(256, 256, 3),
         local_shape=(128, 128, 3)):
    """
    TRAIN A BITCH
    :param params: dict - from argparser the paramaters of erting
    :param global_shape: tuple - assumed RGB - the shape inputted to the net
    :param local_shape: tuple - assumed RGB - local
    """

    # peace of mind
    print('Hyper-parameters:')
    print(params)

    # Set python level verbosity
    tf.logging.set_verbosity(params.verbosity)

    # Set C++ Graph Execution level verbosity  ------- dont know what this is
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[params.verbosity] / 10)

    # we will initialize some saving parameters
    # a note on base_save_dir -> normally, we do not need the "//" after "gs://" because os automatically infers it...
    # however, using google storage, it throws an error without the front slashes.... so keep there there
    base_save_dir = os.path.join('gs://', params.staging_bucketname,
                            params.job_dir)  # use this to construct paths if needed

    ckpt_dir = os.path.join(base_save_dir, 'ckpt/')
    tb_log_dir = os.path.join(base_save_dir, 'tb_logs/')  # save tensorboard logs

    print('base saving directory: {}'.format(base_save_dir))
    print('checkpoint folder: {}'.format(ckpt_dir))
    print('tensorboard logging: {}'.format(tb_log_dir))

    # initialize model_mng and datagenerator
    train_datagen = DataGenerator(params, image_size=global_shape[:-1], local_size=local_shape[:-1])
    # next lets initialize our ModelManager (i.e. the thing that holds the GAN)
    load_ckpt_dir = ckpt_dir if params.load_ckpt else None  # if None - does not search / load any checkpoints (models)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scrope():
        mng = ModelManager(
            optimizer=params.optimizer,
            lr=params.learning_rate,
            alpha=params.alpha,
            load_ckpt_dir=load_ckpt_dir,
            tb_log_dir=tb_log_dir
        )


    # Run the experiment
    time_start = datetime.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    # the actual call to run the experiment
    # mng.run_training_procedure(train_datagen)
    run_job(params=params, model_mng=mng, data_gen=train_datagen, base_save_dir=base_save_dir, strategy=strategy, ckpt_dir=ckpt_dir)

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("")

argument_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialize_hyper_params(argument_parser)


if __name__ == '__main__':

    # we run the experiment with a single call
    main(HYPER_PARAMS)
