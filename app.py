from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

app = Flask(__name__)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

model = tf.keras.models.load_model('model/cifar10_model.h5')

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((32, 32))
    arr = np.array(img).astype('float32') / 255.0
    return arr.reshape(1, 32, 32, 3)

@app.route('/')
def index():
    return render_template('index.html', classes=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    arr = preprocess(file.read())
    preds = model.predict(arr)[0]
    top = int(np.argmax(preds))
    return jsonify({
        'class': CLASS_NAMES[top],
        'confidence': float(preds[top]),
        'scores': {CLASS_NAMES[i]: float(preds[i]) for i in range(10)}
    })

@app.route('/evaluate')
def evaluate():
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical
    (_, _), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.astype('float32') / 255.0
    y_true = y_test.flatten()

    # 10 validation runs on 200 samples each
    results = []
    for run in range(10):
        idx = np.random.choice(len(X_test), 200, replace=False)
        preds = model.predict(X_test[idx])
        y_pred = np.argmax(preds, axis=1)
        acc = np.mean(y_pred == y_true[idx])
        results.append(round(float(acc) * 100, 2))

    # Full confusion matrix
    all_preds = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_true, all_preds)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — CIFAR-10')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png'); buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close()

    overall_acc = float(np.mean(all_preds == y_true)) * 100
    return jsonify({
        'validation_runs': results,
        'mean_accuracy': round(sum(results)/5, 2),
        'test_accuracy': round(overall_acc, 2),
        'confusion_matrix_img': img_b64
    })

if __name__ == '__main__':
    app.run(debug=True)