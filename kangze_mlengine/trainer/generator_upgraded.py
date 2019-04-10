import numpy as np
import cv2
import random as rng
import threading
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from PIL import Image

from google.cloud import storage
from google import auth

try:
    from utils_upgraded import _print

except Exception as e:
    from trainer.utils_upgraded import _print


# might want to throw this in a config file
creds, _ = auth.default()
client = storage.Client()
bucket = client.bucket('lsun-roomsets')


def mask_img(global_size, local_size, hole_min, hole_max):
    # now lets create the random location (aka. X,Y points) where we will apply a mask (aka. erase parts of image)
    # recall that image_size=(256,256) and local_size=(128,128)
    x1 = np.random.randint(0, global_size[0] - local_size[0] + 1)
    y1 = np.random.randint(0, global_size[1] - local_size[1] + 1)
    x2, y2 = np.array([x1, y1]) + np.array(local_size)
    points = [x1, y1, x2, y2]

    # and we also randomly generate width and height of those masks
    w, h = np.random.randint(hole_min, hole_max, 2)
    p1 = x1 + np.random.randint(0, local_size[0] - w)
    q1 = y1 + np.random.randint(0, local_size[1] - h)
    p2 = p1 + w
    q2 = q1 + h
    # now create the array of zeros
    m = np.zeros((global_size[0], global_size[1], 1), dtype=np.uint8)
    # everywhere there should be the mask, make the value one (everywhere else is zero)
    m[q1:q2 + 1, p1:p2 + 1] = 1

    return m, points


def preprocess_img(img, target_size):
    """
    preprocesses the image by cropping (to target_size's aspect ratio), resizing and then normalizing

    NOTE - this funcion uses cv2.INTER_AREA for rescalling, which is prefered way for DOWNSAMPLING (view docs)

    :param img: np array - image type uint8 3 elements, (rows, cols, channels)
    :param target_size: tuple - target img size (rows, cols)
    :return:
    """
    shp = img.shape
    c_ap = float(shp[0] / shp[1])  # current image aspect ratio
    target_ap = float(target_size[0]/target_size[1])  # target aspect ratio height / width * USUALLY 1 (i.e. 256, 256)
    if target_size != shp[0:2]:
        # perform cropping if aspect ratios are not the same
        if c_ap != target_ap:
            # crop to target_size's aspect ratio randomly on the longest dimension of img
            # we crop so the image matches the aspect ratio of self.image_size
            # so, we will crop from the largest dimension
            dim_to_crop = 0 if c_ap > 1 else 1  # find the longest dimension
            x = target_size[dim_to_crop]
            r_not = rng.randint(0, shp[dim_to_crop] - x)  # randomly chosen in between the length of the image and the size_to_match
            # r_not is where we crop from 0 to r_not, r_not + shp[dim_to_crop] is where we pick up cropping to the bottom
            if dim_to_crop == 0:
                # crop height
                output_img = img[r_not:r_not+x, ]
            else:
                # crop width
                output_img = img[:, r_not: r_not+x, ]
        else:
            output_img = img

        # then resize if needed
        if output_img.shape[0] != target_size[0]:
            output_img = cv2.resize(output_img, target_size, interpolation=cv2.INTER_AREA)  # use inter_cubic (preffered for down sampling - generally assumed... ? )
    else:
        # image dimensions match the target dimensions
        output_img = img

    # nomalize
    output_img = output_img / 255.

    return output_img


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class DataGenerator:

    def __init__(self,
             params,
             # batch_size,
             image_size,
             local_size):
        """
        class to iterate through images and yield to generator. Uses a global size and a local (image) size
        Some defaults ----
            bucketname = 'gs://lsun-roomsets'
            input_dir = 'images/bedroom_train/'
            image_size = (256,256)
            local_size = (128,128)
        :param params: object - defined by argparser HYPER_PARAMS outside
        :param image_size: tuple/list (2 elements xy) - size of imgs
        :param local_size: tuple/list (2 elements xy) -
            all images.
        """
        # super(DataGenerator, ).__init__()
        self.params = params
        self.max_img_cnt = self.params.max_img_cnt
        self.image_size = image_size
        self.local_size = local_size
        self.verbosity = params.verbosity
        # self.batch_size = batch_size

    # @threadsafe_generator
    def flow_from_directory(self, hole_min=64, hole_max=128, verbosity=None):
        """
        iterates over img_dir indefinitely and does preprocessing. computes random masks on the fly. loops indefinitely
        :param batch_size: INT - the size of the current batch
        :param hole_min: INT -
        :param hole_max: INT -
        :return:
        """
        while True:
            _print(self.verbosity, 'first FULL LOOP in img generator (i.e. at top of while loop)', ['DEBUG'])
            images, masks, points = [], [], []
            # for now we get max self.count photos and add them to self.img_file_list
            for img_cnt, blob in enumerate(bucket.list_blobs(prefix=self.params.img_dir)):
                # np.random.shuffle(self.img_file_list)
                if self.max_img_cnt and img_cnt >= self.max_img_cnt:
                    print('max image count of {} reached... breaking'.format(self.max_img_cnt))
                    break
                img_url = blob.name
                _print(self.verbosity, 'reading data with fileIO', ['DEBUG'])
                with file_io.FileIO('gs://{}/{}'.format(self.params.bucketname, img_url), 'rb') as f:
                    # and use PIL to convert into an RGB image
                    _print(self.verbosity, 'opening img', ['DEBUG'])
                    img = Image.open(f).convert('RGB')
                    # then convert the RGB image to an array so that cv2 can read it
                    img = np.asarray(img, dtype="uint8")

                    # NOTE the following function randomly crops the image so it has same aspect ratio as image_size, then
                    # it resizes it using cv2 to image_size (this prevents like curved beds etc from warping presumably).
                    img = tf.cast(preprocess_img(img, target_size=self.image_size), tf.float32)

                    # # to look at imgs use - plt.imshow(img_resized[0, :, :, ])  (plt == matplotlib.pyplot)
                    # images.append(img)

                    # get the mask and the bounding box points
                    m, pts = mask_img(self.image_size, self.local_size, hole_min=hole_min, hole_max=hole_max)

                    m, pts = tf.cast(m, dtype=tf.uint8), tf.cast(pts, dtype=tf.uint8)

                    # these are the images with the patches blacked out (i.e. set to zero) - same size as images
                    # WARNING HAPPENS HERE!!!
                    # print('this seems fucked (tf.multiply) in generator')
                    _print(self.verbosity, 'erasing img', ['DEBUG'])
                    erased_img = tf.multiply(img,
                                              tf.cast(tf.subtract(tf.constant(1, dtype=tf.uint8), m),
                                                      dtype=tf.float32))

                    # finally append them to the main tings
                    # masks.append(m)
                    # points.append(pts)

                    # yield erased_imgs, img, m, tf.cast(pts, dtype=tf.int32)
                    _print(self.verbosity, 'yielding img', ['DEBUG'])
                    yield erased_img, img, pts

                    # # yield the batch of data when batch size reached
                    # if len(images) == self.batch_size:
                    #     # this probably wastes some memory needlessly...
                    #     ii, mm, pp = tf.cast(images, tf.float32), tf.cast(masks, dtype=tf.uint8), tf.cast(points, dtype=tf.uint8)
                    #     # probably a better way to do this stupid shit
                    #     images, masks, points = [], [], []
                    #     yield ii, mm, pp
