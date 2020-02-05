import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from loader import Loader

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
exit()
BATCH_SIZE = 256
BUFFER_SIZE = 60000
EPOCHES = 300
OUTPUT_DIR = "img" # The output directory where the images of the generator a stored during training
ROW = 200
COL = 180
CHANNEL = 3

directory = 'img/'
path_a = 'faces94/malestaff/tony/'
path_b = 'faces94/malestaff/voudcx/'
AB_loader = Loader(path_a, path_b)

train_images_a, train_images_b = AB_loader.get_all()
assert train_images_a[0].shape == (ROW, COL, CHANNEL)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images_a, train_images_b)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# ## Generator Network

class Generator(keras.Model):
    
    def __init__(self):
        super().__init__(name='generator')
        #layers
        self.input_layer_a = keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(ROW, COL ,CHANNEL))
        self.input_layer_b = keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(ROW, COL ,CHANNEL))
        self.max_1 = keras.layers.MaxPooling2D()
        self.conv_1 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.max_2 = keras.layers.MaxPooling2D()
        self.conv_2 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.max_3 = keras.layers.MaxPooling2D()
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(units = 128)
        self.leaky_1 =  keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_2 = keras.layers.Dense(units = 128)
        self.leaky_2 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_3 = keras.layers.Dense(units = 256)
        self.leaky_3 = keras.layers.LeakyReLU(alpha = 0.01)
        self.output_layer = keras.layers.Dense(units=ROW * COL * CHANNEL, activation = "tanh")
        self.reshape_output = tf.keras.layers.Reshape(target_shape=(ROW, COL, CHANNEL))
        
    def call(self, input_a, input_b):
        ## Definition of Forward Pass
        h_a = self.input_layer_a(input_a)
        h_b = self.input_layer_b(input_b)
        h_a, h_b = self.max_1(h_a), self.max_1(h_b)
        h_a, h_b = self.conv_1(h_a), self.conv_1(h_b)
        h_a, h_b = self.max_2(h_a), self.max_2(h_b)
        h_a, h_b = self.conv_2(h_a), self.conv_2(h_b)
        h_a, h_b = self.max_3(h_a), self.max_3(h_b)
        f_a, f_b = self.flatten(h_a), self.flatten(h_b)
        x = tf.keras.layers.concatenate([f_a, f_b])
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.dense_2(x)
        x = self.leaky_2(x)
        x = self.dense_3(x)
        x = self.leaky_3(x)
        x = self.output_layer(x)
        x = self.reshape_output(x)
        return x
    
    def generate_noise(self,batch_size, random_noise_size):
        return np.random.uniform(-1,1, size = (batch_size, random_noise_size))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def generator_objective(dx_of_gx):
    # Labels are true here because generator thinks he produces real images. 
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx) 

generator = Generator()
# ## Discriminator Network

class Discriminator(keras.Model):
    def __init__(self):
        super().__init__(name = "discriminator")
        
        #Layers
        self.input_layer = keras.layers.Dense(units = ROW * COL * CHANNEL)
        self.dense_1 = keras.layers.Dense(units = 128)
        self.leaky_1 =  keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_2 = keras.layers.Dense(units = 128)
        self.leaky_2 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_3 = keras.layers.Dense(units = 128)
        self.leaky_3 = keras.layers.LeakyReLU(alpha = 0.01)
        
        self.logits = keras.layers.Dense(units = 1)  # This neuron tells us if the input is fake or real
    def call(self, input_tensor):
          ## Definition of Forward Pass
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.leaky_2(x)
        x = self.leaky_3(x)
        x = self.leaky_3(x)
        x = self.logits(x)
        return x

discriminator = Discriminator()

# ### Objective Function

# In[21]:


def discriminator_objective(d_x, g_z, smoothing_factor = 0.9):
    """
    d_x = real output
    g_z = fake output
    """
    real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor, d_x) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(tf.zeros_like(g_z), g_z) # Each noise we feed in are fakes image --> Because of that labels are 0
    
    total_loss = real_loss + fake_loss
    
    return total_loss

generator_optimizer = keras.optimizers.RMSprop()
discriminator_optimizer = keras.optimizers.RMSprop()


# ## Training Functions
@tf.function()
def training_step(generator: Discriminator, discriminator: Discriminator, images_a:np.ndarray, images_b:np.ndarray , k:int =1, batch_size = 32):
    for _ in range(k):
         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            g_z = generator(images_a, images_b)
            d_x_true = discriminator(images_a) # Trainable?
            d_x_fake = discriminator(g_z) # dx_of_gx

            discriminator_loss = discriminator_objective(d_x_true, d_x_fake)
            # Adjusting Gradient of Discriminator
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) # Takes a list of gradient and variables pairs
            
              
            generator_loss = generator_objective(d_x_fake)
            # Adjusting Gradient of Generator
            gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) 
    
seed = np.random.uniform(-1,1, size = (1, 100)) # generating some noise for the training

# Just to make sure the output directory exists..
import os
if not os.path.exists(directory):
    os.makedirs(directory)

def training(dataset, epoches):
    for epoch in range(epoches):
        for images_a, images_b in dataset: 
            training_step(generator, discriminator, images_a, images_b, batch_size=BATCH_SIZE, k=1)
        ## After ith epoch plot image 
        if (epoch % 50) == 0: 
            fake_image = generator(images_a, images_b)
            print("{}/{} epoches".format(epoch, epoches))
            #plt.imshow(fake_image, cmap = "gray")
            plt.imsave("{}/{}.png".format(OUTPUT_DIR,epoch),fake_image)

training(train_dataset, 100)

# ## Obsolete Training Function
# 
# I tried to implement the training step with the k factor as described in the original paper. I achieved much worse results as with the function above. Maybe i did something wrong?!
# 
@tf.function()
def training_step(generator: Discriminator, discriminator: Discriminator, images:np.ndarray , k:int =1, batch_size = 256):
    for _ in range(k):
        with tf.GradientTape() as disc_tape:
            noise = generator.generate_noise(batch_size, 100)
            g_z = generator(noise)
            d_x_true = discriminator(images) # Trainable?
            d_x_fake = discriminator(g_z) # dx_of_gx

            discriminator_loss = discriminator_objective(d_x_true, d_x_fake, smoothing_factor=0.9)
            # Adjusting Gradient of Discriminator
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) # Takes a list of gradient and variables pairs
    
    with tf.GradientTape() as gen_tape:
        noise = generator.generate_noise(batch_size, 100)
        d_x_fake = discriminator(generator(noise))
        generator_loss = generator_objective(d_x_fake)
        # Adjusting Gradient of Generator
        gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) 
