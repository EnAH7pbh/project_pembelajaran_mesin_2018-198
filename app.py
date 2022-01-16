import os
from flask import Flask, request, render_template
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
import time

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


@app.after_request
def add_header(r):
    # Add headers to both force latest IE rendering engine or Chrome Frame,
    # and also to cache the rendered page for 10 minutes.

    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def result_file():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def cc_predict():
    model = tf.keras.models.load_model('static/asset/places.h5')
    imgweb = request.files["file"]
    img = Image.open(imgweb)
    img = img.convert('RGB')
    img.save("static/uploads/queryImg.jpg")
    data = os.path.join('static/uploads/queryImg.jpg')
    img = Image.open(data)
    img = img.resize((100, 100))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    start = time.time()
    classes = model.predict(img)
    runtimes = np.round(time.time() - start, 4)
    respon_model = [np.round(elem * 100, 2) for elem in classes[0]]
    return predict_result("VGG 19", classes, runtimes, respon_model, 'uploads/queryImg.jpg')


def predict_result(model, result, run_time, probs, img):
    with open('static/asset/classlist', 'rb') as fp:
        classList = pickle.load(fp)
    labels = classList[np.argmax(result)]
    idx_pred = probs.index(max(probs))
    return render_template('/result_select.html', labels=labels,
                           probs=probs, model=model, pred=idx_pred,
                           run_time=run_time, img=img)


if __name__ == '__main__':
    app.run()
