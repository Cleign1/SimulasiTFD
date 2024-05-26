# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_image, test_label) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0

    test_image = test_image.reshape(10000, 28, 28, 1)
    test_image = test_image / 255.0

    class Custom_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > 0.89):
                print("\nValidation Accuracy >= 90%")
                print("\nTraining Selesai")
                self.model.stop_training = True
                
    callback = Custom_callback()
    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3), activation='relu',input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units= 128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    # COMPILE MODEL HERE
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    model.fit(training_images, training_labels, validation_data=(test_image, test_label),epochs=10,
              callbacks=callback)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
