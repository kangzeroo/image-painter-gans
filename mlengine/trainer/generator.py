from tensorflow.python.lib.io import file_io
from google.cloud import storage
import cv2
from PIL import Image
import numpy as np



# Data generator that will feed in our data batch by batch
class DataGenerator(object):
    # initialize by retreiving the photos
    def __init__(self, BUCKET_NAME, INPUT_DIR, IMAGE_SIZE, LOCAL_SIZE):
        # BUCKET_NAME = 'lsun-roomsets'
        # INPUT_DIR = 'images/bedroom_train/'
        # IMAGE_SIZE = (256,256)
        # LOCAL_SIZE = (128,128)
        self.IMAGE_SIZE = IMAGE_SIZE
        self.LOCAL_SIZE = LOCAL_SIZE
        self.reset()
        self.img_file_list = []
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        # for now we get max self.count photos and add them to self.img_file_list
        for blob in bucket.list_blobs(prefix=INPUT_DIR):
            self.img_file_list.append(blob.name)

    def __len__(self):
        return len(self.img_file_list)

    # we also track the preprocessed images, points, and masks
    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    # iterates over self.img_file_list and does preprocessing
    def flow(self, BATCH_SIZE, BUCKET_NAME, HOLE_MIN=64, HOLE_MAX=128):
        np.random.shuffle(self.img_file_list)
        for idx, img_url in enumerate(self.img_file_list):
            # we use tf...file_io.FileIO to grab the file
            with file_io.FileIO("gs://" + BUCKET_NAME + "/" + img_url, 'rb') as f:
                # and use PIL to convert into an RGB image
                img = Image.open(f).convert('RGB')
                # then convert the RGB image to an array so that cv2 can read it
                img = np.asarray(img, dtype="uint8")
                # resize images
                img_resized = cv2.resize(img, self.IMAGE_SIZE)[:,:,::-1]
                # take a look at the images
                # cv2.imshow(f'image_{idx}_resized', img_resized)
                # cv2.waitKey(0)
                # cv2.destroyWindow(f'image_{idx}_resized')
                # add the resized photo to self.images
                self.images.append(img_resized)

                # now lets create the random location (aka. X,Y points) where we will apply a mask (aka. erase parts of image)
                # recall that IMAGE_SIZE=(256,256) and LOCAL_SIZE=(128,128)
                x1 = np.random.randint(0, self.IMAGE_SIZE[0] - self.LOCAL_SIZE[0] + 1)
                y1 = np.random.randint(0, self.IMAGE_SIZE[1] - self.LOCAL_SIZE[1] + 1)
                x2, y2 = np.array([x1, y1]) + np.array(self.LOCAL_SIZE)
                self.points.append([x1,y1,x2,y2])
                # and we also randomly generate width and height of those masks
                w, h = np.random.randint(HOLE_MIN, HOLE_MAX, 2)
                p1 = x1 + np.random.randint(0, self.LOCAL_SIZE[0] - w)
                q1 = y1 + np.random.randint(0, self.LOCAL_SIZE[1] - h)
                p2 = p1 + w
                q2 = q1 + h
                # now create the array of zeros
                m = np.zeros((self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 1), dtype=np.uint8)
                # everywhere there should be the mask, make the value one (everywhere else is zero)
                m[q1:q2 + 1, p1:p2 + 1] = 1
                # finally append it to the self.masks
                self.masks.append(m)

                # yeild the batch of data when batch size reached
                if len(self.images) == BATCH_SIZE:
                    images = np.asarray(self.images, dtype=np.float32) / 255
                    points = np.asarray(self.points, dtype=np.int32)
                    masks = np.asarray(self.masks, dtype=np.float32)
                    self.reset()
                    yield images, points, masks

def createGenerator(BUCKET_NAME, INPUT_DIR, IMAGE_SIZE, LOCAL_SIZE):
    return DataGenerator(
        BUCKET_NAME,
        INPUT_DIR,
        IMAGE_SIZE,
        LOCAL_SIZE
    )
