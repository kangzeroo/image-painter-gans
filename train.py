# playable version of DataGenerator()
class FuckAround():
    # list the bucket and directory within
    bucketname = 'gs://lsun-roomsets'
    directory = 'images/bedroom_val/'

    # loop through all the files and view first X images (count)
    count = 0
    max_count = 10
    # store the raw img urls here
    img_urls = []
    for blob in bucket.list_blobs(prefix=directory):
        if count >= max_count:
            break
        print(blob.name)
        count += 1
        img_urls.append(blob.name)

    # store the resized images here
    images = []
    points = []
    masks = []

    # CONSTANTS
    # mask size limits
    hole_min = 64
    hole_max = 128
    # batch limits
    batch_size = 5
    max_batches = 3
    batch_count = 0
    # image sizes
    image_size = (256, 256)
    local_size = (128, 128)

    for idx, img_url in enumerate(img_urls):
        # we use tf...file_io.FileIO to grab the file
        with file_io.FileIO(f'{bucketname}/{img_url}', 'rb') as f:
            # and use PIL to convert into an RGB image
            img = Image.open(f).convert('RGB')
            # then convert the RGB image to an array so that cv2 can read it
            img = np.asarray(img, dtype="uint8")
            # resize images
            img_resized = cv2.resize(img, image_size)[:, :, ::-1]
            # take a look at the images
            # cv2.imshow(f'image_{idx}_resized', img_resized)
            # cv2.waitKey(0)
            # cv2.destroyWindow(f'image_{idx}_resized')
            # add the resized photo to self.images
            images.append(img_resized)
            print(f'{idx}. Processing {img_url}')

            # now lets create the random points where we will apply a mask (erase parts of image)
            # recall that image_size=(256,256) and local_size=(128,128)
            x1 = np.random.randint(0, image_size[0] - local_size[0] + 1)
            y1 = np.random.randint(0, image_size[1] - local_size[1] + 1)
            x2, y2 = np.array([x1, y1]) + np.array(local_size)
            points.append([x1, y1, x2, y2])

            # and we also randomly generate width and height of those masks
            w, h = np.random.randint(hole_min, hole_max, 2)
            p1 = x1 + np.random.randint(0, local_size[0] - w)
            q1 = y1 + np.random.randint(0, local_size[1] - h)
            p2 = p1 + w
            q2 = q1 + h
            # now create the array of zeros
            m = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)
            # everywhere there should be the mask, make the value one (everywhere else is zero)
            m[q1:q2 + 1, p1:p2 + 1] = 1
            # finally append it to the self.masks
            masks.append(m)

            # print the batch of data when batch size reached
            if len(images) == batch_size:
                #                 print(np.array(images).shape)
                #                 print(np.array(points).shape)
                #                 print(np.array(masks).shape)
                inputs = np.asarray(images, dtype=np.float32) / 255
                points = np.asarray(points, dtype=np.int32)
                masks = np.asarray(masks, dtype=np.float32)

                # reset
                images = []
                points = []
                masks = []
                batch_count += 1

            if batch_count > max_batches:
                break


class DataGenerator(object):
    # initialize by retreiving the photos
    def __init__(self, bucketname, input_dir, image_size, local_size):
        # bucketname = 'gs://lsun-roomsets'
        # input_dir = 'images/bedroom_train/'
        # image_size = (256,256)
        # local_size = (128,128)
        self.image_size = image_size
        self.local_size = local_size
        self.reset()
        self.img_file_list = []
        # for now we get max self.count photos and add them to self.img_file_list
        for blob in bucket.list_blobs(prefix=input_dir):
            self.img_file_list.append(blob.name)

    def __len__(self):
        return len(self.img_file_list)

    # we also track the preprocessed images, points, and masks
    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    # iterates over self.img_file_list and does preprocessing
    def flow(self, batch_size, hole_min=64, hole_max=128):
        np.random.shuffle(self.img_file_list)
        for idx, img_url in enumerate(self.img_file_list):
            # we use tf...file_io.FileIO to grab the file
            with file_io.FileIO(f'{bucketname}/{img_url}', 'rb') as f:
                # and use PIL to convert into an RGB image
                img = Image.open(f).convert('RGB')
                # then convert the RGB image to an array so that cv2 can read it
                img = np.asarray(img, dtype="uint8")
                # resize images
                img_resized = cv2.resize(img, self.image_size)[:, :, ::-1]
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
                    images = np.asarray(self.images, dtype=np.float32) / 255
                    points = np.asarray(self.points, dtype=np.int32)
                    masks = np.asarray(self.masks, dtype=np.float32)
                    self.reset()
                    yield images, points, masks


if __name__ == '__main__':
    # hyperparameters
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)
    batch_size = 96
    epochs = 500000
    g_epochs = int(epochs * 0.18)  # should be 90k on generator
    d_epochs = int(epochs * 0.02)  # should be 10k on discriminator
    alpha = 0.0004

    batch_count = 0

    # input/output directories
    bucketname = "gs://lsun-roomsets"
    output_dir = "outputs/"
    input_dir = "images/bedroom_val/"

    alpha = 0.0004

    full_img = Input(shape=global_shape)
    clip_img = Input(shape=local_shape)
    mask = Input(shape=(global_shape[0], global_shape[1], 1))
    ones = Input(shape=(global_shape[0], global_shape[1], 1))
    clip_coords = Input(shape=(4,), dtype='int32')

    gen_brain, gen_model = full_gen_layer(full_img, mask, ones)
    disc_brain, disc_model = full_disc_layer(global_shape, local_shape, full_img, clip_coords)

    print(gen_brain)
    print(disc_brain)

    print(gen_model)
    print(disc_model)

    # the final brain
    disc_model.trainable = False
    connected_disc = Model(inputs=[full_img, clip_coords], outputs=disc_model)
    connected_disc.name = 'Connected-Discrimi-Hater'
    print(connected_disc)

    brain = Model(inputs=[full_img, mask, ones, clip_coords],
                  outputs=[gen_model, connected_disc([gen_model, clip_coords])])
    brain.compile(loss=['mse', 'binary_crossentropy'],
                  loss_weights=[1.0, alpha], optimizer=optimizer)
    brain.summary()
    view_models(brain, 'summaries/brain.png')


    # data generator
    train_datagen = DataGenerator(bucketname, input_dir, input_shape[:2], local_shape[:2])

    # train over time
    for epoch in range(epochs):
        # progress bar visualization (comment out in ML Engine)
        progbar = generic_utils.Progbar(len(train_datagen))
        for images, points, masks in train_datagen.flow(batch_size):
            # and the matrix of ones that we depend on in the neural net to inverse masks
            mask_inv = np.ones((len(images), input_shape[0], input_shape[1], 1))
            # generate the inputs (images)
            generated_img = gen_brain.predict([images, masks, mask_inv])
            # generate the labels
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            # the gen and disc losses
            g_loss = 0.0
            d_loss = 0.0

            # we must train the neural nets seperately, and then together
            # train generator for 90k epochs
            if epoch < g_epochs:
                # set the gen loss
                g_loss = gen_brain.train_on_batch([images, points], valid)
            # train discriminator alone for 90k epochs
            # then train disc + gen for another 400k epochs. Total of 500k
            else:
                # throw in real unedited images with label VALID
                d_loss_real = disc_brain.train_on_batch([images, points], valid)
                # throw in A.I. generated images with label FAKE
                d_loss_fake = disc_brain.train_on_batch([generated_img, points], fake)
                # combine and set the disc loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if epoch >= tc + td:
                    # train the entire brain
                    g_loss = brain.train_on_batch([images, masks, mask_inv, points], [images, valid])
                    # and update the generator loss
                    g_loss = g_loss[0] + alpha * g_loss[1]
            # progress bar visualization (comment out in ML Engine)
            progbar.add(images.shape[0], values=[("Disc Loss: ", d_loss), ("Gen mse: ", g_loss)])
            batch_count += 1
            # save the generated image
            last_img = generated_img[0]
            last_img[:, :, 0] = last_img[:, :, 0] * 255
            last_img[:, :, 1] = last_img[:, :, 1] * 255
            last_img[:, :, 2] = last_img[:, :, 2] * 255
            dreamt_image = Image.fromarray(last_img.astype(int), 'RGB')
            dreamt_image.save(f"outputs/images/batch_{batch_count}_image.png")

        gen_brain.save(f"outputs/models/batch_{batch_count}_generator.h5")
        disc_brain.save(f"outputs/models/batch_{batch_count}discriminator.h5")

