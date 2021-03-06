import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
from PIL import Image
import cv2
import pdb

from google.cloud import storage
from google import auth

# might want to throw this in a config file
creds, _ = auth.default()
client = storage.Client()
bucket = client.bucket('lsun-roomsets')


class DataGenerator(object):
    # initialize by retreiving the photos
    def __init__(self,
             params,
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
        self.params = params
        self.max_img_cnt = self.params.max_img_cnt
        self.image_size = image_size
        self.local_size = local_size
        self.reset()
        self.img_file_list = []
        self.images = []
        self.masks = []
        self.points = []

    def __len__(self):
        return len(self.img_file_list)

    def reset(self):
        # we also track the preprocessed images, points, and masks
        self.images = []
        self.masks = []
        self.points = []

    def flow(self, batch_size, hole_min=64, hole_max=128):
        """
        iterates over self.img_file_list and does preprocessing. computes random masks on the fly.
        :param batch_size: INT - the size of the current batch
        :param hole_min: INT -
        :param hole_max: INT -
        :return:
        """
        # for now we get max self.count photos and add them to self.img_file_list
        for img_cnt, blob in enumerate(bucket.list_blobs(prefix=self.params.img_dir)):
            np.random.shuffle(self.img_file_list)
            if self.max_img_cnt and img_cnt >= self.max_img_cnt:
                print('max image count of {} reached... breaking'.format(self.max_img_cnt))
                break
            img_url = blob.name
            with file_io.FileIO('gs://{}/{}'.format(self.params.bucketname, img_url), 'rb') as f:
                # and use PIL to convert into an RGB image
                img = Image.open(f).convert('RGB')
                # then convert the RGB image to an array so that cv2 can read it
                img = np.asarray(img, dtype="uint8")
                # resize images
                img_resized = cv2.resize(img, self.image_size)[:, :, ::-1]  # this is giving me a not found error fml...
                # take a look at the images
                # cv2.imshow(f'image_{idx}_resized', img_resized)
                # cv2.waitKey(0)
                # cv2.destroyWindow(f'image_{idx}_resized')
                # add the resized photo to self.images
                self.images.append(img_resized)

                # now lets create the random location (aka. X,Y points) where we will apply a mask (aka. erase parts of image)
                # recall that image_size=(256,256) and local_size=(128,128)
                x1 = np.random.randint(0, self.image_size[0] - self.local_size[0] + 1)
                y1 = np.random.randint(0, self.image_size[1] - self.local_size[1] + 1)
                x2, y2 = np.array([x1, y1]) + np.array(self.local_size)
                self.points.append([x1, y1, x2, y2])

                # and we also randomly generate width and height of those masks
                w, h = np.random.randint(hole_min, hole_max, 2)
                p1 = x1 + np.random.randint(0, self.local_size[0] - w)
                q1 = y1 + np.random.randint(0, self.local_size[1] - h)
                p2 = p1 + w
                q2 = q1 + h
                # now create the array of zeros
                m = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
                # everywhere there should be the mask, make the value one (everywhere else is zero)
                m[q1:q2 + 1, p1:p2 + 1] = 1
                # finally append it to the self.masks
                self.masks.append(m)

                # yeild the batch of data when batch size reached
                if len(self.images) == batch_size:
                    images = np.asarray([a / 255 for a in self.images])
                    masks = self.masks
                    points = self.points
                    self.reset()
                    # dataset = tf.data.Dataset.from_tensor_slices(
                    #     (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
                    #      tf.cast(mnist_labels, tf.int64)))
                    # dataset = dataset.shuffle(1000).batch(32)
                    yield images, masks, points
