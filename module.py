import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np

glcm_distances = [1]
glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm_properties = ['contrast', 'homogeneity', 'energy', 'correlation']
hsv_properties = ['hue', 'saturation', 'value']


def extract_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    glcm_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(glcm_image, distances=glcm_distances,
                        angles=glcm_angles, symmetric=True, normed=True)

    hsv_features = []
    for property_name in hsv_properties:
        property_value = hsv_image[:, :,
                                   hsv_properties.index(property_name)].ravel()
        hsv_features.extend([np.mean(property_value), np.std(property_value)])

    glcm_features = []
    for property_name in glcm_properties:
        property_value = graycoprops(glcm, property_name).ravel()
        glcm_features.extend([np.mean(property_value), np.std(property_value)])

    features = hsv_features + glcm_features
    return features
