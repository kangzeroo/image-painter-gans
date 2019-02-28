import os
from tensorflow.python.lib.io import file_io
from keras.utils import plot_model


def view_models(model, filename):
    """
    causes a crash... with pyplot in gc - could probably just add to setup...
    :param model:
    :param filename:
    :return:
    """
    plot_model(model, to_file=filename, show_shapes=True)



def copy_file_to_gcs(BUCKET_NAME, OUTPUT_DIR, FILE_PATH):
  """
  copy a file to gcs
  :param BUCKET_NAME: str - name of bucket
  :param OUTPUT_DIR: str - the output directory inside the bucket
  :param FILE_PATH: str - any other path
  :return:
  """
  with file_io.FileIO(FILE_PATH, mode='rb') as input_f:
    with file_io.FileIO('gs://{}/{}/{}'.format(BUCKET_NAME, OUTPUT_DIR, FILE_PATH), mode='w+') as output_f:
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


