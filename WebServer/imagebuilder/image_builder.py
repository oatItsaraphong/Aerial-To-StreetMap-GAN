import tensorflow as tf
from matplotlib import pyplot as plt
from image_model_adapter import ImageModelAdapter



BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
PATHIN = "../cacheimages/aerialimage"
PATHOUT = "../cacheimages/streetgen"
CHECKPOINT_DIR = '../../checkpoint'

class ImageBuilder:
    def __init__(self, version) -> None:

        image_adapter = ImageModelAdapter(version)
        image_model = image_adapter.getModel()
        
        checkpoint_dir = CHECKPOINT_DIR + "/v" + str(version)
        #print(checkpoint_dir)
        self.pathOut = PATHOUT  + "v" + str(version)

        self.generator = image_model.Generator()
        self.discriminator = image_model.Discriminator()
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


    def discriminator_loss_2(self, disc_real_output, disc_generated_output):
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
        
        prediction = model(test_input, training=True)
        #print(prediction[0])
        self.saveImage(filePath, prediction[0])
        return True

    #Public User
    def generate_image(self, name):
        fileIn = PATHIN + "/" + name
        fileOut = self.pathOut + "/" +name

        test_input = tf.data.Dataset.list_files(fileIn)
        test_input= test_input.map(self.load_image_obj)
        test_input = test_input.shuffle(BUFFER_SIZE)
        test_input = test_input.batch(BATCH_SIZE)

        for example_input in test_input:
            self.generate_images(fileOut,self.generator, example_input)

        return True
        
  
# end ImageBuilder class


