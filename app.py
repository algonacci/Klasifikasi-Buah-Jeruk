import os
import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

import module as md

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/img/'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


class_names = ["Greening", "Blackspot", "Canker", "Fresh"]
knn = joblib.load('knn_model.pkl')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=["GET", 'POST'])
def predict():
    if request.method == "POST":
        image = request.files['image']
        if image and allowed_file(image.filename):
            image.save(os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename)))
            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
            image_cv = cv2.imread(image_path)
            image_cv = cv2.resize(image_cv, (64, 64))
            features = md.extract_features(image_cv)
            prediction = knn.predict([features])
            predicted_class = prediction[0].item()
            result = class_names[predicted_class]

            # Add the description, prevention, and treatment based on the predicted class
            description = ""
            prevention = ""
            treatment = ""

            if predicted_class == 0:  # Greening
                description = "Greening adalah penyakit yang disebabkan oleh bakteri dan umumnya menyerang tanaman jeruk. Gejalanya termasuk daun berwarna kuning muda, penurunan pertumbuhan, dan pucuk tanaman yang mati."
                prevention = "Beberapa tindakan pencegahan Greening meliputi memperhatikan kebersihan kebun, memotong dan membuang tanaman yang terinfeksi, serta penggunaan bibit yang bebas penyakit."
                treatment = "Sayangnya, tidak ada pengobatan yang efektif untuk Greening. Tindakan yang dapat dilakukan adalah memantau dan mengendalikan serangga vektor penyakit serta merawat tanaman dengan baik agar tetap sehat."

            elif predicted_class == 1:  # Blackspot
                description = "Blackspot adalah penyakit yang umum terjadi pada tanaman jeruk. Gejalanya berupa bintik-bintik hitam pada daun dan buah, yang kemudian dapat menyebabkan penurunan kualitas dan produksi tanaman."
                prevention = "Tindakan pencegahan Blackspot meliputi pemangkasan dan pemotongan bagian tanaman yang terinfeksi, penggunaan fungisida yang sesuai, serta memastikan kelembaban udara yang baik di sekitar tanaman."
                treatment = "Pengobatan Blackspot melibatkan penggunaan fungisida yang efektif dan penerapan tindakan pengendalian penyakit yang tepat waktu. Penting untuk memantau dan merawat tanaman secara teratur."

            elif predicted_class == 2:  # Canker
                description = "Canker adalah penyakit yang disebabkan oleh jamur atau bakteri dan dapat menginfeksi batang dan cabang tanaman jeruk. Gejalanya berupa lesi atau luka pada kulit tanaman."
                prevention = "Beberapa tindakan pencegahan Canker meliputi pemangkasan dan pembuangan cabang yang terinfeksi, penggunaan fungisida atau antibakteri yang direkomendasikan, serta memastikan sanitasi yang baik di kebun."
                treatment = "Pengobatan Canker melibatkan penggunaan fungisida atau antibakteri yang sesuai, pemangkasan tanaman untuk menghilangkan jaringan yang terinfeksi, dan pemeliharaan kebersihan yang baik."

            elif predicted_class == 3:  # Fresh
                description = "Tanaman Anda sehat dan tidak terkena penyakit."
                prevention = ""
                treatment = ""

            return render_template("prediction.html", result=result, image_path=image_path, description=description, prevention=prevention, treatment=treatment)

        else:
            return render_template("index.html", error="Silahkan upload gambar dengan format JPG")

    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run()
