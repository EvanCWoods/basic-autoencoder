import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.datasets
from tensorflow.keras.datasets import mnist

fashion_mnist = tensorflow.keras.datasets.fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

def raw_data_info(train_data, test_data):
    print('Raw data info:')
    print('Train data min: ', train_data.min())
    print('Train data max: ', train_data.max())
    print('Test data min: ', test_data.min())
    print('Test data max: ', test_data.max())
    print('Train data shape: ', train_data.shape)
    print('Test data shape: ', test_data.shape)
    print('Train data dtype: ', train_data.dtype)
    print('Test data dtype: ', test_data.dtype)
    print('</--DONE--\>')

raw_data_info(train_data=train_data, test_data=test_data)


# Function to preprocess the data
def preprocess(train_data, test_data):
    train_data = train_data / 255 
    print('Train data min: ', train_data.min())
    print('Train data max: ', train_data.max())
    test_data = test_data / 255
    print('Test data min: ', test_data.min())
    print('Test dat max: ', test_data.max())

preprocess(train_data=train_data, test_data=test_data)



tf.random.set_seed(42)
encoder = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=[28,28]),
  tf.keras.layers.Dense(256, activation='selu'),
  tf.keras.layers.Dense(128, activation='selu'),
  tf.keras.layers.Dense(64, activation='selu'),
  tf.keras.layers.Dense(30, activation='selu'),
])

decoder = tf.keras.Sequential([
  tf.keras.layers.Dense(256, activation='selu', input_shape=[30]),
  tf.keras.layers.Dense(128, activation='selu'),
  tf.keras.layers.Dense(64, activation='selu'),
  tf.keras.layers.Dense(28 * 28, activation='sigmoid'),
  tf.keras.layers.Reshape([28, 28])
])

stacked_ae = tf.keras.models.Sequential([encoder, decoder])
stacked_ae.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam())

history = stacked_autoencoder.fit(train_data, train_data, epochs=20,
                         validation_data=(test_data, test_data))

def plot_image(image):
    plt.imshow(image)
    plt.axis('off')

def show_recostruction(model, n_images):
    reconstruction = model.predict(test_data[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1+ image_index)
        plot_image(test_data[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstruction[image_index])

show_recostruction(model=stacked_autoencoder, n_images=5)

