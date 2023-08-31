import cv2
import numpy as np

def detect_color_object(image, lower_bound, upper_bound):
    # Konversi gambar ke format HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Buat mask untuk warna yang ditentukan
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Temukan kontur objek dalam mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar bounding box di sekitar objek yang terdeteksi
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image

# Baca gambar dari file
image = cv2.imread('gambar/1.jpg')

# Definisikan rentang warna yang ingin dideteksi (dalam format HSV)
lower_color = np.array([100, 100, 100])  # Ganti dengan nilai yang sesuai
upper_color = np.array([130, 255, 255])  # Ganti dengan nilai yang sesuai

# Deteksi objek berwarna dalam gambar
result_image = detect_color_object(image, lower_color, upper_color)

# Tampilkan hasilnya
cv2.imshow('Color Object Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()