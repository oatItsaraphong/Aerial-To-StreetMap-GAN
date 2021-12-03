import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 

  w = tf.shape(image)[1]
  w = w // 2
  satellite_image = image[:, w:, :]
  aerial_image = image[:, :w, :]

  # Convert both images to float32 tensors
  satellite_image = tf.cast(satellite_image, tf.float32)
  aerial_image = tf.cast(aerial_image, tf.float32)

  return aerial_image, satellite_image

## image manipulation
def resize(input_img, real_img, height, width):
    input_img = tf.image.resize(input_img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_img = tf.image.resize(real_img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_img, real_img

def random_crop(input_img, real_img):
    stacked_img = tf.stack([input_img, real_img], axis=0)
    cropped_img = tf.image.random_crop(stacked_img, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_img[0], cropped_img[1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_img, real_img):

    #without jitter ----------------------
    input_img, real_img = resize(input_img, real_img, IMG_HEIGHT, IMG_WIDTH)
    #-----------------------------
    # with Jitter ----------------
    #resize
    # input_img, real_img = resize(input_img, real_img, 286, 286)
    # #randome crop to 256
    # input_img, real_img = random_crop(input_img, real_img)
    ###  ----------------------------------
  

    if tf.random.uniform(()) > 0.5:
        #random mirror
        input_img = tf.image.flip_left_right(input_img)
        real_img = tf.image.flip_left_right(real_img)
    
    return input_img, real_img

def load_image_train(image_file):
  input_image, real_image= load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image= load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image





## === Generator ====

def downsample(filters, size, apply_batchnorm=True, apply_dropout=False, dropout_value=0.5, strides_num=2):
  #add apply drop out
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strides_num, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(dropout_value))

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False,dropout_value=0.5, strides_num=2):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides_num,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(dropout_value))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  ## revert the dropput - pix2pix  - street to real 
  ## move kernal size to 9
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])
  k_size = 4

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4, apply_dropout=True, dropout_value=0.5),  # (batch_size, 64, 64, 128)
    downsample(256, 4, apply_dropout=True, dropout_value=0.5),  # (batch_size, 32, 32, 256)
    downsample(512, 4, apply_dropout=True, dropout_value=0.5),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4),  # (batch_size, 2, 2, 1024)
    upsample(512, 4),  # (batch_size, 4, 4, 1024)
    upsample(512, 4),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
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



def generator_loss(disc_generated_output, gen_output, target):

  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


## === Discriminator

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  k_size = 4
  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, k_size, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, k_size)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, k_size)(down2)  # (batch_size, 32, 32, 256)

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



def discriminator_loss(disc_real_output, disc_generated_output):

  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


## === Training ===
@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
  #example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      #generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)


## Start process

## data input config
train_path = "./merge16/train"
test_path = "./merge16/test"
val_path = "./merge16/validation"

## image config
BUFFER_SIZE = 2500

BATCH_SIZE = 1

IMG_HEIGHT = 256
IMG_WIDTH = 256


# Generator config
OUTPUT_CHANNELS = 3
LAMBDA = 100

## Get data
train_dataset = tf.data.Dataset.list_files(str(train_path + '/*.jpeg'))
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
  test_dataset = tf.data.Dataset.list_files(str(test_path + '/*.jpeg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(val_path + '/*.jpeg'))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


## Set up model

generator = Generator()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator = Discriminator()

# set optimizer
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# checkpoint
checkpoint_dir = './training_checkpointsv30'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



log_dir="./logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fitv30/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


fit(train_dataset, test_dataset, steps=40000)

print("============= End Training =================")