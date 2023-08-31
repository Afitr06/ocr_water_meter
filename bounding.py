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

# Deteksi objek berwarna biru
lower_blue = np.array([100, 100, 100])  # Nilai-nilai bisa disesuaikan
upper_blue = np.array([130, 255, 255])  # Nilai-nilai bisa disesuaikan
result_image_blue = detect_color_object(image.copy(), lower_blue, upper_blue)

# Deteksi objek berwarna merah
lower_red = np.array([68, 100, 100])   # Nilai-nilai bisa disesuaikan
upper_red = np.array([10, 211, 217])  # Nilai-nilai bisa disesuaikan
result_image_red = detect_color_object(image.copy(), lower_red, upper_red)

# Gabungkan hasil deteksi berwarna biru dan merah
result_image_combined = cv2.addWeighted(result_image_blue, 1, result_image_red, 1, 0)

# Tampilkan hasilnya
cv2.imshow('Color Object Detection', result_image_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
