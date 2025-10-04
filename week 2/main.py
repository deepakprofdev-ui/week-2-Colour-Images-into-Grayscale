from src.preprocess import load_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import test_model

def main():
    print("🚀 Loading Data...")
    x_train, y_train, x_test, y_test = load_data()

    print("🧠 Building Model...")
    model = build_model()
    model.summary()

    print("🎯 Training Model...")
    train_model(model, x_train, y_train)

    print("📊 Evaluating Model...")
    evaluate_model(model, x_test, y_test)

    print("🖼️ Testing Model Output...")
    test_model(model, x_test)

if __name__ == "__main__":
    main()
