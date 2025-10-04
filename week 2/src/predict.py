import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def test_model(model, x_test):
    """Display model predictions vs traditional grayscale."""
    preds = model.predict(x_test[:5])
    
    for i in range(5):
        plt.figure(figsize=(8,3))
        plt.subplot(1,3,1)
        plt.imshow(x_test[i])
        plt.title("Original")

        plt.subplot(1,3,2)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title("Model Output")

        plt.subplot(1,3,3)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title("Grayscale Prediction")

        plt.show()
