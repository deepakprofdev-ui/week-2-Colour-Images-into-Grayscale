import os

def train_model(model, x_train, y_train, save_path="models/grayscale_model.h5"):
    """Train the model and save it."""
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")
