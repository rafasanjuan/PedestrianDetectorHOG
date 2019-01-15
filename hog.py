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



# GAMMA/COLOR NORMALIZATION
#Compute the normalize gamma and colour
img = np.power(img, 0.5, dtype=np.float32)


# GRADIENT COMPUTATION
# Compute centered horizontal and vertical gradients with no smoothing
kernel = np.asarray([-1, 0, 1])
kernel_vacio = np.asarray([1])

gradientsx = cv2.sepFilter2D(img, -1, kernel,kernel_vacio)
showImgAndWait("Gradientes", gradientsx)

gradientsy = cv2.sepFilter2D(img, -1,kernel_vacio, kernel)
showImgAndWait("Gradientes", gradientsy)



# SPATIAL / ORIENTATION BINNING
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
#print(img.shape)
#print(max_angle.shape)
showImgAndWait("angulos",max_angle)

def genHistogram(angles, magnitudes):
    histogram = np.zeros(9, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            # Find the two nearests bins.
            bin1 = int( angles[i, j] // 20 )
            bin2 = int(( (angles[i, j] // 20) + 1) % 9)

            # calculate the proportional vote for the two affected angles
            prop = (angles[i, j] - (bin1 * 20)) / 20

            # calculate the value vote for the two affected bin
            vote1 = (1 - prop) * magnitudes[i, j]
            vote2 =  prop * magnitudes[i, j]

            histogram[bin1] += vote1
            histogram[bin2] += vote2

    return histogram

def genCells(angles, magnitudes, cell_size):
    cells = []
    for i in range(0, np.shape(angles)[0], cell_size):
        row = []
        for j in range(0, np.shape(angles)[1], cell_size):
            histogram = genHistogram(angles[i:i + cell_size, j:j + cell_size],
                    magnitudes[i:i + cell_size, j:j + cell_size])
            row.append(histogram)
            
        cells.append(row)

    return np.array(cells, dtype=np.float32)

cell = genCells(max_angle,max_magnitude,6)
print(cell.shape)


