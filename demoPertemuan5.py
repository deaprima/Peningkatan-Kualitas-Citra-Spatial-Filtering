import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk Histogram Equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Fungsi untuk Spatial Smoothing menggunakan Mean Filter
def spatial_smoothing(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    smoothed_image = cv2.filter2D(image, -1, kernel)
    return smoothed_image

# Fungsi untuk Spatial Smoothing menggunakan Median Filter
def spatial_smoothing_median(image, kernel_size=3):
    smoothed_image = cv2.medianBlur(image, kernel_size)
    return smoothed_image

# Fungsi untuk Spatial Sharpening menggunakan Sobel Filter
def spatial_sharpening(image):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    grad_x = cv2.filter2D(image, -1, sobel_kernel_x)
    grad_y = cv2.filter2D(image, -1, sobel_kernel_y)

    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)

    magnitude = cv2.magnitude(grad_x, grad_y)
    
    return magnitude

# Fungsi untuk Spatial Sharpening menggunakan Prewitt Filter
def prewitt_sharpening(image):
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
    
    grad_x = cv2.filter2D(image, -1, prewitt_kernel_x)
    grad_y = cv2.filter2D(image, -1, prewitt_kernel_y)
    
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    return magnitude

# Fungsi untuk Spatial Sharpening menggunakan Roberts Filter
def roberts_sharpening(image):
    roberts_kernel_x = np.array([[1, 0], [0, -1]], np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], np.float32)

    grad_x = cv2.filter2D(image, -1, roberts_kernel_x)
    grad_y = cv2.filter2D(image, -1, roberts_kernel_y)
    
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    return magnitude


image = cv2.imread('lena.webp', cv2.IMREAD_GRAYSCALE)

equalized_image = histogram_equalization(image)
smoothed_image = spatial_smoothing(image, kernel_size=3)
median_smoothed_image = spatial_smoothing_median(image, kernel_size=3)
sobel_sharpened_image = spatial_sharpening(image)
prewitt_sharpened_image = prewitt_sharpening(image)
roberts_sharpened_image = roberts_sharpening(image)

cv2.imwrite('equalized_image.jpg', equalized_image)
cv2.imwrite('smoothed_image.jpg', smoothed_image)
cv2.imwrite('median_smoothed_image.jpg', median_smoothed_image)
cv2.imwrite('sobel_sharpened_image.jpg', sobel_sharpened_image)
cv2.imwrite('prewitt_sharpened_image.jpg', prewitt_sharpened_image)
cv2.imwrite('roberts_sharpened_image.jpg', roberts_sharpened_image)

plt.figure(figsize=(10, 10))

plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(smoothed_image, cmap='gray')
plt.title("Spatial Smoothing (Mean Filter)")
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(median_smoothed_image, cmap='gray')
plt.title("Spatial Smoothing (Median Filter)")
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(sobel_sharpened_image, cmap='gray')
plt.title("Spatial Sharpening (Sobel)")
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(prewitt_sharpened_image, cmap='gray')
plt.title("Spatial Sharpening (Prewitt)")
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(roberts_sharpened_image, cmap='gray')
plt.title("Spatial Sharpening (Roberts)")
plt.axis('off')

plt.tight_layout()
plt.show()
