import tensorflow as tf
import os
from matplotlib import pyplot as plt
import zerorpc

OUTPUT_CHANNELS = 3
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
PATHIN = "../aerialimage/"
PATHOUT = "../streetgen/"


checkpoint_dir = '../../checkpoint/v15'

class ImageBuilder:
    def __init__(self) -> None:
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    #Generator
    def Generator(self):
        ## revert the dropput - pix2pix  - street to real 
        ## move kernal size to 9
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        k_size = 9

        down_stack = [
            self.downsample(64, k_size, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            self.downsample(128, k_size),  # (batch_size, 64, 64, 128)
            self.downsample(256, k_size),  # (batch_size, 32, 32, 256)
            self.downsample(512, k_size, apply_dropout=True),  # (batch_size, 16, 16, 512)
            self.downsample(512, k_size),  # (batch_size, 8, 8, 512)
            self.downsample(512, k_size, apply_dropout=True),  # (batch_size, 4, 4, 512)
            self.downsample(512, k_size, apply_dropout=True),  # (batch_size, 2, 2, 512)
            self.downsample(512, k_size, apply_dropout=True),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, k_size),  # (batch_size, 2, 2, 1024)
            self.upsample(512, k_size),  # (batch_size, 4, 4, 1024)
            self.upsample(512, k_size),  # (batch_size, 8, 8, 1024)
            self.upsample(512, k_size),  # (batch_size, 16, 16, 1024)
            self.upsample(256, k_size),  # (batch_size, 32, 32, 512)
            self.upsample(128, k_size),  # (batch_size, 64, 64, 256)
            self.upsample(64, k_size),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def downsample(self, filters, size, apply_batchnorm=True, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.LeakyReLU())
        return result

    #Discriminator
    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        k_size = 9
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = self.downsample(64, k_size, False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, k_size)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, k_size)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, k_size, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, k_size, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    #Image Load and save
    def load(self,image_file):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        satellite_image = image
        satellite_image = tf.cast(satellite_image, tf.float32)

        return satellite_image


    def load_image_obj(self, image_file):
        input_image= self.load(image_file)
        input_image = tf.image.flip_left_right(input_image)
        input_image = tf.image.flip_left_right(input_image)
        input_image= self.normalize(input_image)
        return input_image

    def normalize(self, input_image):
        input_image = (input_image / 127.5) - 1
        return input_image

    def saveImage(self, file, data):
        plt.subplot(1, 1, 1)
        #plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(data * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig(file, bbox_inches='tight', transparent="True", pad_inches=0)

    def generate_images(self,filePath,model, test_input, tar=None):
        prediction = model(test_input)
        self.saveImage(filePath, prediction[0])
        return True

    #Public User
    def generate_image(self, name):
        fileIn = PATHIN + name
        fileOut = PATHOUT + name

        test_input = tf.data.Dataset.list_files(fileIn)
        test_input= test_input.map(self.load_image_obj)
        test_input = test_input.shuffle(BUFFER_SIZE)
        test_input = test_input.batch(BATCH_SIZE)

        for example_input in test_input:
            print(example_input.shape)
            self.generate_images(fileOut,self.generator, example_input)

        return True
  
# end ImageBuilder class


# obj = ImageBuilder()
# print("==================ImageBuild==================")
# obj.generate_image("abc.png")
# print("==================ImageBuild==================")
# obj.generate_image("abc1.png")
# print("==================ImageBuild==================")
# obj.generate_image("abc2.png")


print("===Server Start===========")
server = zerorpc.Server(ImageBuilder())
server.bind("tcp://0.0.0.0:4242")
print("Server running on port 4242")
server.run()


## zerorpc