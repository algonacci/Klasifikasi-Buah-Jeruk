import cv2
import numpy as np
from flask import Flask, request, jsonify
from skimage.feature import graycomatrix, graycoprops
import joblib

app = Flask(__name__)

# Mendefinisikan ukuran gambar yang digunakan
image_size = (64, 64) # bisa 64, 256, dll

class_names = ["grenning", "blackspot", "canker", "fresh"]

# Mendefinisikan metode GLCM
glcm_distances = [1]
glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm_properties = ['contrast', 'homogeneity', 'energy', 'correlation']

# Mendefinisikan metode HSV
hsv_properties = ['hue', 'saturation', 'value']

# Memuat model KNN
knn = joblib.load('knn_model.pkl')

# Mendefinisikan fungsi untuk ekstraksi fitur dari gambar
def extract_features(image):
    # Melakukan ekstraksi fitur dari gambar
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    glcm_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(glcm_image, distances=glcm_distances, angles=glcm_angles, symmetric=True, normed=True)
    
    hsv_features = []
    for property_name in hsv_properties:
        property_value = hsv_image[:,:,hsv_properties.index(property_name)].ravel()
        hsv_features.extend([np.mean(property_value), np.std(property_value)])
    
    glcm_features = []
    for property_name in glcm_properties:
        property_value = graycoprops(glcm, property_name).ravel()
        glcm_features.extend([np.mean(property_value), np.std(property_value)])
    
    features = hsv_features + glcm_features
    return features

@app.route('/index')
def index():
    return "Ini adalah endpoint untuk klasifikasi buah jeruk menggunakan Flask."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, image_size)
    
    features = extract_features(image)
    
    # Melakukan prediksi menggunakan model
    prediction = knn.predict([features])
    
    # Mengembalikan hasil prediksi dalam bentuk JSON response
    result = {'class': class_names[prediction[0].item()]}
    return jsonify({
        "result": result
    })

if __name__ == '__main__':
    app.run()
