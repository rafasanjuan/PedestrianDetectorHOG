# PedestrianDetectorHOG
Implementación del algoritmo descrito en https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf, para la detección
de peatones en imágenes.

Para ello se extrae el descriptor expuesto de todas las imágenes ejemplo, en las cuales tendremos casos positivos y negativos.
Con ello se entrenará una SVM para tener un clasificador que determine si en la imágen hay un peaton. Para el uso en casos reales
con una imagen dada, se recorre la imagen con una ventana y se determina con el clasificador si en dicha facción de la imágen hay 
un peaton
