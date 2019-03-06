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
    # NOTE -- if the file directly after the bucket is not created (i.e. theos_jobs) initially,
    # this throws an error - carefull baby
    with file_io.FileIO(save_path, 'wb') as f:
        print('\nsaving image at {}\n'.format(save_path))
        img_data.save(f, "PNG")
