import numpy as np
from skimage.metrics import structural_similarity as ssim

def evaluate_model(model, x_test, y_test):
    """Evaluate the model using SSIM metric."""
    preds = model.predict(x_test[:100])  # sample 100 images
    scores = []
    for i in range(len(preds)):
        s = ssim(y_test[i].squeeze(), preds[i].squeeze(), data_range=1.0)
        scores.append(s)
    
    print(f"Average SSIM: {np.mean(scores):.4f}")
