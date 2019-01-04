import cv2
import numpy as np

def showImgAndWait(titulo, img):
    cv2.imshow(titulo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("image.png", 1)
cv2.imshow("Titulo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#Compute the normalize gamma and colour
img = np.power(img, 0.2, dtype=np.float32)


# Compute centered horizontal and vertical gradients with no smoothing
kernel = np.asarray([-1, 0, 1])
kernel_vacio = np.asarray([1])

gradientsx = cv2.sepFilter2D(img, -1, kernel,kernel_vacio)
showImgAndWait("Gradientes", gradientsx)

gradientsy = cv2.sepFilter2D(img, -1,kernel_vacio, kernel)
showImgAndWait("Gradientes", gradientsy)

# Calculate the magnitudes and angles of the gradients
magnitude, angle = cv2.cartToPolar(gradientsx, gradientsy, angleInDegrees=True)



# If the image is in color, we pick the channel with the biggest value as the 
# intensity of that pixel.
intensity = np.argmax(magnitude, axis=2)


x, y = np.ogrid[:intensity.shape[0], :intensity.shape[1]]
max_angle = angle[x, y, intensity]
max_magnitude = magnitude[x, y, intensity]

showImgAndWait("Gradientes", max_magnitude)
#showImgAndWait("Gradientes", max_angles)

# 0-360 angle to 0-180
max_angle = (360 - max_angle) % 180

def make_cells(angles, magnitudes, cell_size):
    cells = []
    for i in range(0, np.shape(angles)[0], cell_size):
        row = []
        for j in range(0, np.shape(angles)[1], cell_size):
            row.append(np.array(
                histogram(angles[i:i + cell_size, j:j + cell_size], magnitudes[i:i + cell_size, j:j + cell_size]),
                dtype=np.float32))
        cells.append(row)

    return np.array(cells, dtype=np.float32)