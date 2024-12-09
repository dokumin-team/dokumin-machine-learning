import tensorflow as tf
import cv2
import numpy as np

# Fungsi untuk memuat model .h5
def load_h5_model(h5_path):
    model = tf.keras.models.load_model(h5_path)
    return model

# Fungsi untuk melakukan preprocessing pada gambar
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca dari path: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Fungsi untuk menjalankan prediksi menggunakan model .h5
def predict_image(h5_model_path, image_path, label_map):
    # Memuat model .h5
    model = load_h5_model(h5_model_path)

    # Preprocessing gambar
    input_data = preprocess_image(image_path)

    # Jalankan prediksi
    output_data = model.predict(input_data)
    predicted_label = np.argmax(output_data)
    confidence = np.max(output_data)

    # Tampilkan hasil
    label_name = [k for k, v in label_map.items() if v == predicted_label][0]
    print(f"Prediksi: {label_name} (Confidence: {confidence:.2f})")


# Label mapping (sama dengan yang digunakan saat melatih model)
label_map = {'document': 0, 'KTP': 1, 'KK': 2, 'SIM': 3}

# Path model .h5 dan gambar
h5_model_path = "model_klasifikasi.h5"
image_path = "./dataset/test/00836816_png.rf.5ff458f0e88623400e47569e94c47910.jpg"

# Jalankan prediksi
predict_image(h5_model_path, image_path, label_map)
