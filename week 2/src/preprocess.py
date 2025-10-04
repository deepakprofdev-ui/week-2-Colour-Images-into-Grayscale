import tensorflow as tf
import numpy as np

def rgb2gray(images):
    """Convert RGB images to grayscale."""
    return np.expand_dims(np.dot(images[..., :3], [0.299, 0.587, 0.114]), axis=-1)

def load_data():
    """Load CIFAR-10 dataset and prepare grayscale targets."""
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Create grayscale targets
    y_train = rgb2gray(x_train)
    y_test = rgb2gray(x_test)

    print(f"Training data: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Testing data: {x_test.shape}, Labels: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test
