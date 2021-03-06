import numpy as np
from tensorflow.python.lib.io import file_io
from PIL import Image
import cv2
import random as rng

from google.cloud import storage
from google import auth

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

    def flow(self, batch_size, hole_min=64, hole_max=128):
        """
        iterates over img_dir and does preprocessing. computes random masks on the fly.
        :param batch_size: INT - the size of the current batch
        :param hole_min: INT -
        :param hole_max: INT -
        :return:
        """
        images, masks, points = [], [], []
        # for now we get max self.count photos and add them to self.img_file_list
        for img_cnt, blob in enumerate(bucket.list_blobs(prefix=self.params.img_dir)):
            # np.random.shuffle(self.img_file_list)
            if self.max_img_cnt and img_cnt >= self.max_img_cnt:
                print('max image count of {} reached... breaking'.format(self.max_img_cnt))
                break
            img_url = blob.name
            with file_io.FileIO('gs://{}/{}'.format(self.params.bucketname, img_url), 'rb') as f:
                # and use PIL to convert into an RGB image
                img = Image.open(f).convert('RGB')
                # then convert the RGB image to an array so that cv2 can read it
                img = np.asarray(img, dtype="uint8")

                # NOTE the following function randomly crops the image so it has same aspect ratio as image_size, then
                # it resizes it using cv2 to image_size (this prevents like curved beds etc from warping presumably).
                img = preprocess_img(img, target_size=self.image_size)

                # to look at imgs use - plt.imshow(img_resized[0, :, :, ])  (plt == matplotlib.pyplot)
                images.append(img)

                # get the mask and the bounding box points
                m, pts = mask_img(self.image_size, self.local_size, hole_min=hole_min, hole_max=hole_max)

                # finally append them to the main tings
                masks.append(m)
                points.append(pts)

                # yield the batch of data when batch size reached
                if len(images) == batch_size:
                    # this probably wastes some memory needlessly...
                    ii, mm, pp = np.asarray(images).copy(), masks.copy(), points.copy()
                    # probably a better way to do this stupid shit
                    images, masks, points = [], [], []
                    yield ii, mm, pp
