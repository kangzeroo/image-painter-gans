import argparse
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from keras.utils import generic_utils
from PIL import Image


# enable eager execution......
# QUESTION - do we need to call this in model.py also for example???
# tf.compat.v1.enable_eager_execution()
# tf.executing_eagerly()


try:
    from model_upgraded import ModelManager
    from generator_upgraded import DataGenerator
    from utils_upgraded import save_img, extract_roi_imgs, save_model, log_scalar

except Exception as e:
    # # cloud - multi
    from trainer.model_upgraded import ModelManager
    from trainer.generator_upgraded import DataGenerator
    from trainer.utils_upgraded import save_img, extract_roi_imgs, save_model, log_scalar


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
        default=2  # currently 25 throws memory errors...... NEED TO INCREASE THIS BABY (use 20 for now)
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""\
            Maximum number of training data epochs on which to train.
            If both --train-size and --num-epochs are specified,
            --train-steps will be: (train-size/train-batch-size) * num-epochs.\
            """,
        default=50000,
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
        default=1,
        help="""\
            if params.save_shit is true, this determines how often as modulo(epoch, params.epoch_save_frequency)""",
        type=int,
    )
    args_parser.add_argument(
        '--job-dir',
        # default="gs://temp/outputs",
        default="distributed_attempt1",
        type=str,
    )
    args_parser.add_argument(
        '--img-dir',
        # default="gs://temp/outputs",
        default="images/bedroom_train",
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
        '--save-shit',
        # default="gs://temp/outputs",
        help="""\
            True or False to save everything - ckpts, examples, etc.""",
        default=False,
        type=bool,
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
        default='Adadelta',
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


def main(params,
         global_shape=(256, 256, 3),
         local_shape=(128, 128, 3),
         g_epochs=2,
         d_epochs=2):
    """
    TRAIN A BITCH
    :param params: dict - from argparser the paramaters of erting
    :param global_shape: tuple - assumed RGB - the shape inputted to the net
    :param local_shape: tuple - assumed RGB - local
    """

    # peace of mind
    print('Hyper-parameters:')
    print(params)

    # multi gpu
    # strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1", "/device:GPU:2"])
    # strategy = tf.compat.v1.distribute.MirroredStrategy(num_gpus=3)

    # tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print('found {} machines'.format(mirrored_strategy.num_replicas_in_sync))

    # Set python level verbosity
    tf.compat.v1.logging.set_verbosity(params.verbosity)

    # multiple cpu shit?
    # with tf.variable_creator_scope("queue"):
    #     q = tf.queue.FIFOQueue(capacity=5, dtypes=tf.float32)  # enqueue 5 batches
    #     # We use the "enqueue" operation so 1 element of the queue is the full batch
    #     enqueue_op = q.enqueue(x_input_data)
    #     numberOfThreads = 1
    #     qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    #     tf.train.add_queue_runner(qr)
    #     input = q.dequeue()  # It replaces our input placeholder

    # coordinator for multiple cpus?
    # coord = tf.train.Coordinator()
    #
    # q = tf.queue.FIFOQueue(capacity=3, dtypes=tf.float32)


    # Set C++ Graph Execution level verbosity  ------- dont know what this is
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[params.verbosity] / 10)

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

    # Run the experiment
    time_start = datetime.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    # with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
    #     for i in range(0, 3):  # number of gpus
    #         with tf.device('/gpu:%d' % i):
    #             with tf.name_scope('%s_%d' % ('tower_bitch', i)) as scope:
    # initialize model_mng and datagenerator

    # construct the dataset with the mirrored strategy
    train_datagen = DataGenerator(params, image_size=global_shape[:-1], local_size=local_shape[:-1])

    # next lets initialize our ModelManager (i.e. the thing that holds the GAN)
    load_ckpt_dir = ckpt_dir if params.load_ckpt else None  # if None - does not search / load any checkpoints (models)

    # # the actual call to run the experiment
    # # mng.run_training_procedure(train_datagen)
    # run_job(params=params, model_mng=mng, data_gen=train_datagen, base_save_dir=base_save_dir, ckpt_dir=ckpt_dir, tb_log_dir=tb_log_dir)
    tb_logger = tf.summary.create_file_writer(tb_log_dir)

    # train baby bitch
    if g_epochs != int(params.num_epochs * 0.18) or int(params.num_epochs * 0.02):
        print('###### WARN - generator or discriminator epochs are not default as paper!!!')

    # where we save weights (NOT FULLY IMPLEMENTED / VALIDATED)
    model_save_dir = os.path.join(base_save_dir, 'models/')

    generated_imgs = None  # redundant... but to stop warning in IDE
    # init_epoch = model_mng.epoch.numpy()
    prog_cap = params.steps_per_epoch * params.train_batch_size if params.max_img_cnt is None else params.max_img_cnt

    with mirrored_strategy.scope():

        # we need to create the iterator / generator using tensorflows Dataset - this enable multi-gpus etc
        # do so within the scope of mirrored_strategy -- NOTE right now, the train_datagen itself has its threadsafe turned to false....
        ds = tf.data.Dataset.from_generator(
            train_datagen.flow_from_directory,
            (tf.float32, tf.float32, tf.uint8, tf.int32),
            # output_shapes=(tf.TensorShape([params.train_batch_size, ] + list(global_shape)),
            #                tf.TensorShape([params.train_batch_size, ] + list(global_shape[:-1]) + [1, ]),
            #                tf.TensorShape([params.train_batch_size, 4])))
            output_shapes=(tf.TensorShape(list(global_shape)),
                           tf.TensorShape(list(global_shape)),
                           tf.TensorShape(list(global_shape[:-1]) + [1, ]),
                           tf.TensorShape([4]))
        ).batch(params.train_batch_size)
        # ds = ds.batch()
        # ds.map()

        input_iterator = mirrored_strategy.make_dataset_iterator(ds)
        # get output from data generator like:
        # z, zz, zzz = input_iterator.get_next() for example.

        # create models etc inside the strategy's scope. This ensures that any variables created with the model
        # and optimizer are mirrored variables.
        model_mng = ModelManager(
            # strategy=mirrored_strategy,
            optimizer=params.optimizer,
            lr=params.learning_rate,
            alpha=params.alpha,
            load_ckpt_dir=load_ckpt_dir,
        )

        def train_generator(inputs):
            # for each step, we get the data from the generator
            erased_imgs, images, _ = inputs

            # train generator
            g_loss = model_mng.train_gen(erased_imgs, images)

            return g_loss

        def train_discriminator(inputs):

            erased_imgs, images, points = inputs

            generated_imgs = distributed_predict_generator(erased_imgs)

            roi_imgs_real, roi_imgs_fake = extract_roi_imgs(images, points), extract_roi_imgs(generated_imgs, points)

            # generate the labels
            valid = np.ones((params.train_batch_size, 1))
            fake = np.zeros((params.train_batch_size, 1))

            # train the discriminator
            d_loss_real = model_mng.train_disc(images, roi_imgs_real, valid)
            d_loss_fake = model_mng.train_disc(generated_imgs, roi_imgs_fake, fake)

                # # # combine and set the disc loss
                # d_loss = tf.multiply(tf.add(d_loss_real, d_loss_fake), 0.5)
                # # with tf.name_scope('discriminator_loss'):
                # #     variable_summaries(d_loss)
            # # return combined_loss, g_loss, d_loss

            return d_loss_real, d_loss_fake

        def train_brain(inputs):

            erased_imgs, images, points = inputs

            fake = np.zeros((params.train_batch_size, 1))

            # # tf.summary.scalar('discriminator_loss', d_loss)
            # train the entire brain (note this only updates the generator - but uses joint loss gen + disc)
            combined_loss, g_loss = model_mng.train_full_brain(erased_imgs, images, points, fake)

            return combined_loss, g_loss

        @tf.function
        def distributed_train_generator():
            return mirrored_strategy.experimental_run(train_generator, input_iterator)

        @tf.function
        def distributed_predict_generator(imgs):
            return model_mng.gen_model(imgs, training=False)  # FOR SOME REASON PREDICTING WITH TRAINING=FALSE GIVES NANS

        @tf.function
        def distributed_train_discriminator():
            return mirrored_strategy.experimental_run(train_discriminator, input_iterator)

        @tf.function
        def distributed_train_brain():
            return mirrored_strategy.experimental_run(train_brain, input_iterator)

        input_iterator.initialize()
        for epoch in range(params.num_epochs):
            print('epoch: ' + str(epoch))
            # if epoch < g_epochs:
            if False:

                g_loss = distributed_train_generator()

            else:

                d_loss_real, d_loss_fake = distributed_train_discriminator()

                if epoch >= g_epochs + d_epochs:
                    dicks, cunts = distributed_train_brain()

            # print(train_step(batch_size=params.train_batch_size,
            #                  strategy=mirrored_strategy,
            #                  epoch=epoch,
            #                  model_mng=model_mng,
            #                  g_epochs=0, d_epochs=2))
            # # increment the self.epoch  -> we need to do this so that the checkpoint is accurate....
            # with mirrored_strategy.scope():
            #     model_mng.epoch.assign_add(1)  # note this might be stupid --- can lead to desynchronization ...
            # # might consider just setting model_mng.epoch = tensor(epoch) for example.
            # if epoch % params.epoch_save_frequency == 0 and epoch > 0 and params.save_shit:
            #     # write to tensorboard
            #     with tb_logger.as_default():
            #         tf.summary.scalar('generator_loss', g_loss.numpy(), step=epoch)
            #         tf.summary.scalar('discriminator_loss', d_loss.numpy(), step=epoch)
            #         tf.summary.scalar('combined_loss', combined_loss.numpy(), step=epoch)
            #
            #     # save check_point - THESE GET PICKED UP IF SPECIFIED
            #     print('\nsaving checkpoint {}\n'.format(ckpt_dir))
            #     model_mng.checkpoint.save(ckpt_dir)
            #     if params.save_weights:
            #         # NOTE - this is not full implemented or tested ~~~~~~~~
            #         # save the model weights and architecture... these dont get used because we pick up checkpoints FYI
            #         # note this is pretty slow tbh
            #         save_model(
            #             model=model_mng.gen_model,
            #             save_path=os.path.join(model_save_dir, 'generator_epoch_{}'.format(epoch))
            #         )
            #         save_model(
            #             model=model_mng.disc_model,
            #             save_path=os.path.join(model_save_dir, 'discriminator_epoch_{}'.format(epoch))
            #         )
            #     # save a generated image for peace of mind
            #     if generated_imgs is not None:
            #         last_img = generated_imgs[0]
            #         last_img *= 255
            #         dreamt_image = Image.fromarray(np.asarray(last_img, dtype='uint8'), 'RGB')
            #         output_image_path = os.path.join(base_save_dir, 'images', 'epoch_{}_image.png'.format(epoch))
            #         save_img(save_path=output_image_path, img_data=dreamt_image)

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
    main(
        HYPER_PARAMS,
        g_epochs=int(HYPER_PARAMS.num_epochs * 0.18),
        d_epochs=int(HYPER_PARAMS.num_epochs * 0.02),
    )
