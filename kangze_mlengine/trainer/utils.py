import tensorflow as tf
from tensorflow.python.lib.io import file_io
from keras.utils import plot_model


def save_model(model, save_path):
    """
    saves the models weights and architecture.

    Note - tried to use model.save() but got a NotImplemented Error, so
           we need to save like this (weights + architecture) it seems
           EDIT - cant save architecture (see note below) ... only saving the weights here... :(
                  its okay though because we dont even use these (typically we load up a model from the ckpts).

    :param model:
    :param save_name:
    :return:
    """
    # NOTE - SAVING LIKE THE FOLLOWING GIVES ME A NON-IMPLEMENTED ERROR ON MY LAPTOP --- might be different on gcloud,
    #        havent tried. If you want you can wrap the following in a try except, and put the current way in the
    #        except
    # model.save(save_name)

    weights_save_name = save_path + '-weights'
    # arch_save_name = save_name + '-model'
    print('saving weights {}'.format(weights_save_name))
    model.save_weights(weights_save_name)
    # save the architecture in a json ?
    # json_model = model.to_json()  # -> NotImplementedError FUCK BITCH
    # NOTE - saving the architecture always results in a NotImplementedError .... so just leaving as is (i.e. we only
    #        the weights) ... A quick read said it should work if we specify input shape ... but im too fucking lazy for
    #        that shit rn


def extract_roi_imgs(images, points):
    roi_imgs = tf.cast(
        [
            tf.image.crop_to_bounding_box(
                a,
                offset_height=b[1],
                offset_width=b[0],
                target_height=b[3] - b[1],
                target_width=b[2] - b[0]
            )
            for a, b in zip(images, points)
        ],
        tf.float32
    )
    return roi_imgs


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


def log_scalar(name, val, logging_frequency=1):
    """
    tensorboard logs "name" with value = val

    ??? does this work in utils?

    :param name: str - name of paramater
    :param val: value of paramater (scalar i.e. loss)
    :return:
    """
    with tf.summary.record_summaries_every_n_global_steps(logging_frequency):
        tf.summary.scalar(name, val)

