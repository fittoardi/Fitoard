# Model-Warna-pada-citra

## Pengujian Model Warna Pada Citra

```python
# Image Processing Techniques on Google Colab

from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Upload image file
uploaded = files.upload()

# Membaca gambar yang diunggah
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fungsi untuk menampilkan gambar asli dan hasil proses

def show_images(title1, image1, title2, image2):

    plt.figure(figsize=(12,6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.show()

# Menampilkan Gambar Asli
show_images('Gambar Asli (Grayscale)', image_gray, 'Gambar Asli (RGB)', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 1. Citra Negative
negative_image = 255 - image_gray
show_images('Gambar Asli', image_gray, 'Citra Negative', negative_image)

# 2. Transformasi Log
c = 255 / np.log(1 + np.max(image_gray))
log_image = c * np.log(1 + image_gray.astype(np.float32))
log_image = np.uint8(log_image)
show_images('Gambar Asli', image_gray, 'Transformasi Log', log_image)

# 3. Transformasi Power Law (Gamma Correction)
gamma = 0.5  # Ubah nilai gamma untuk hasil berbeda
power_law_image = np.array(255 * (image_gray / 255) ** gamma, dtype='uint8')
show_images('Gambar Asli', image_gray, 'Transformasi Power Law', power_law_image)

# 4. Histogram Equalization
equalized_image = cv2.equalizeHist(image_gray)
show_images('Gambar Asli', image_gray, 'Histogram Equalization', equalized_image)

# 5. Histogram Normalization
normalized_image = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
show_images('Gambar Asli', image_gray, 'Histogram Normalization', normalized_image)

# 6. Konversi RGB ke HSI
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsi = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
show_images('Gambar Asli (RGB)', image_rgb, 'Konversi RGB ke HSI (Hue Component)', image_hsi[:,:,0])

# Menentukan thresholding
threshold_value = 120
_, threshold_image = cv2.threshold(image_hsi[:,:,2], threshold_value, 255, cv2.THRESH_BINARY)
show_images('Gambar Asli (Intensity Component)', image_hsi[:,:,2], 'Thresholding pada Komponen Intensity', threshold_image)
```
GAMBAR ORIGINAL PENGUJIAN : London

![download](https://github.com/user-attachments/assets/e66e33e8-f8d2-43de-80cf-a7bce3453982)

![download](https://github.com/user-attachments/assets/0e49154c-d8eb-43d5-8fd5-34349b424889)

![download](https://github.com/user-attachments/assets/3155c7a5-d8ac-4ae8-800a-67aa45e413e0)

![download](https://github.com/user-attachments/assets/2522ca6c-5d4f-42cf-8270-0b2490913cd6)

![download](https://github.com/user-attachments/assets/c5b596a9-278d-49f8-bdd7-ee90d7073e7f)

![download](https://github.com/user-attachments/assets/97f040fc-478c-4bf2-8f04-b83670000a11)

![download](https://github.com/user-attachments/assets/ad140d9a-e68f-4a9e-823f-b43446f26d55)

![download](https://github.com/user-attachments/assets/3d67d60e-6fb8-444a-9f92-5655041c36b6)



