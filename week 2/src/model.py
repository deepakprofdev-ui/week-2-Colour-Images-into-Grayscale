from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, InputLayer

def build_model():
    """Build and return CNN model for grayscale conversion."""
    model = Sequential([
        InputLayer(input_shape=(32, 32, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        Conv2D(1, (3,3), activation='sigmoid', padding='same')  # Grayscale output
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model
