import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf


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
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename("queryImg.jpg")
            data = os.path.join('static/uploads/', 'queryImg.jpg')
            file.save(data)
            return redirect(url_for('result_file'))
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def result_file():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def cc_predict():
    with open('static/asset/classlist', 'rb') as fp:
        classList = pickle.load(fp)
    model = tf.keras.models.load_model('static/asset/places.h5')
    if request.method == 'POST':
        filename = request.form.get('input_image')
        if filename and allowed_file(filename):
            img = Image.open(filename)
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            img.save("static/uploads/queryImg.jpg")
            data = os.path.join('static/uploads/queryImg.jpg')
            img = Image.open(data)
            img = img.resize((100, 100))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            img = np.vstack([img])
            classes = model.predict(img)
            result = classList[np.argmax(classes)]
    return render_template('/index.html', prediction_text=result)


if __name__ == '__main__':
    app.run()
