
import argparse
from datetime import datetime

from trainer.job import run_training_job

# Specify settings for this training execution
def initialize_job_settings(args_parser):
    args_parser.add_argument(
        '--verbosity',
        help='Set logging level',
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

# Specify where the folders and files for input and output are
def initialise_file_locations(args_parser):
    # load in training data folder
    args_parser.add_argument(
        '--train-files',
        help='Specify the gs://path/to/training/data/',
        nargs='+',
        required=False
    )
    # load in evaluation data folder
    args_parser.add_argument(
        '--eval-files',
        help='Specify the gs://path/to/evaluation/data/',
        nargs='+',
        required=False
    )
    # load in job run directory
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        type=str,
        required=False
    )
    # reuse past job dir? deletes old job dir if False
    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""\
            Flag to decide if the model checkpoint should
            be re-used from the job-dir. If False then the
            job-dir will be deleted"""
    )
    return args_parser.parse_args()

# Specify hyper-parameters to training the model
def initialise_hyper_params(args_parser):
    # args_parser.add_argument(
    #     '--num-epochs',
    #     help="""\
    #     Maximum number of training data epochs on which to train.
    #     If both --train-size and --num-epochs are specified,
    #     --train-steps will be: (train-size/train-batch-size) * num-epochs.\
    #     """,
    #     default=100,
    #     type=int,
    # )
    return args_parser.parse_args()


def main():
    print('============ Google ML Engine ============')
    print('Training with job settings:')
    print('')
    print(JOB_PARAMS)
    print('')
    print('-------')
    print('Training with folder paths:')
    print('')
    print(FILE_PARAMS)
    print('')
    print('-------')
    print('')
    print('Training with model hyper-parameters:')
    print('')
    print(HYPER_PARAMS)
    print('')
    print('-------')
    print('')
    time_start = datetime.utcnow()
    print("Training job started at {}".format(time_start.strftime("%H:%M:%S")))
    print('')
    print('==============================================')

    run_training_job(
        JOB_PARAMS = JOB_PARAMS,
        FILE_PARAMS = FILE_PARAMS,
        HYPER_PARAMS = HYPER_PARAMS
    )

    print('==============================================')
    print('')
    time_end = datetime.utcnow()
    print("Training job ended at {}".format(time_start.strftime("%H:%M:%S")))
    print('')
    print('-------')
    print('')
    time_elapsed = time_end - time_start
    print("Training job elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print('')
    print('-------')
    print('')
    print('')
    print('')
    print('============ <END> Google ML Engine </END> ============')

args_parser = argparse.ArgumentParser()
JOB_PARAMS = initialize_job_settings(args_parser)
FILE_PARAMS = initialise_file_locations(args_parser)
HYPER_PARAMS = initialise_hyper_params(args_parser)

if __name__ == '__main__':
    main()
