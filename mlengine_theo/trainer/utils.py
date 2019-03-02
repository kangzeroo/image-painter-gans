from tensorflow.python.lib.io import file_io
from keras.utils import plot_model
# import pdb  # for debug


def view_models(model, filename):
    """
    causes a crash... with pyplot in gc - could probably just add to setup...
    :param model:
    :param filename:
    :return:
    """
    plot_model(model, to_file=filename, show_shapes=True)


def copy_file_to_gcs(bucket_name, output_dir, file_path):

    """
    copy a file to gcs
    :param bucket_name: str - name of bucket
    :param output_dir: str - the output directory inside the bucket
    :param file_path: str - any other path
    :return:
    """
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO('gs://{}/{}/{}'.format(bucket_name, output_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def save_img(save_path, img_data):

    """
    save an image to the save_path
    :param save_path: str path - to save destination
    :param img_data: img pil - img to be saved
    :return:
    """

    with file_io.FileIO(save_path, 'wb') as f:
        print('\nsaving image at {}\n'.format(save_path))
        img_data.save(f, "PNG")


def initialize_hyper_params(args_parser):

    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Is this ok in utils?

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
        default=20  # currently 25 throws memory errors...... NEED TO INCREASE THIS BABY
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""\
            Maximum number of training data epochs on which to train.
            If both --train-size and --num-epochs are specified,
            --train-steps will be: (train-size/train-batch-size) * num-epochs.\
            """,
        default=500,
        type=int,
    )
    args_parser.add_argument(
        '--gen-loss',
        help="""\
            The loss function for generator\
            """,
        default='mse',
        type=str,
    )
    args_parser.add_argument(
        '--disc-loss',
        help="""\
            The loss function for discriminator\
            """,
        default='binary_crossentropy',
        type=str,
    )
    args_parser.add_argument(
        '--alpha',
        default=0.0004,
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
        default=2,
        type=int,
    )
    args_parser.add_argument(
        '--job-dir',
        # default="gs://temp/outputs",
        default="output_BEEFY",
        type=str,
    )
    args_parser.add_argument(
        '--img-dir',
        # default="gs://temp/outputs",
        default="images/bedroom_val",
        type=str,
    )
    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""\
            Flag to decide if the model checkpoint should
            be re-used from the job-dir. If False then the
            job-dir will be deleted"""
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
        default=0.1,
        type=float
    )
    # Estimator arguments
    args_parser.add_argument(
        '--max-img-cnt',
        help="Number of maximum images to look at. Set to None if you"
             "want the whole dataset. Primarily used for testing purposes.",
        default=None,
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
