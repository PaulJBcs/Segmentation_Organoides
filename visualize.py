import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

# Dossier contenant les masques
tiff_path = input("entrez le fichier au format TIFF :")  # Remplace par le chemin de ton dossier

# Charger, redimensionner et extraire les points blancs
points_x, points_y, points_z = [], [], []
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    return open_cv_image
def tiff_to_list(path):
    tiff_image = Image.open(path)

    # Initialiser une liste pour stocker les images
    images_list = []

    # Parcourir toutes les pages (frames) du fichier TIFF
    try:
        while True:
            # Ajouter chaque frame à la liste
            images_list.append(tiff_image.copy())
            
            # Aller à la page suivante
            tiff_image.seek(tiff_image.tell() + 1)
    except EOFError:
        # Quand on atteint la fin du fichier TIFF, une EOFError est levée
        pass
    return images_list
taille = (550,540)
images_list = tiff_to_list(tiff_path)
n = len(images_list)
for z in range(n):  # De mask0.png à mask552.png
    

    # Charger l'image et redimensionner à 128x128
    mask = pil_to_cv2(images_list[z])
    resized_mask = cv2.resize(mask, taille)  # Redimensionner

    # Trouver les indices des pixels blancs dans l'image redimensionnée
    y, x = np.where(resized_mask == 255)
    points_x.extend([11*e for e in x])
    points_y.extend([11*e for e in y])
    points_z.extend([4.5*z] * len(x))  # Ajouter l'indice de la couche (z) pour chaque point
    
print(len(x),len(y))


# Vérification : Nombre total de points détectés
print(f"Nombre total de points blancs : {len(points_x)}")

# Création de la figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


maxx = max(points_x)
maxz = max(points_z)
# Visualisation des points avec les nouvelles valeurs de z
ax.scatter(points_x, points_y, points_z, c='red', s=1)
# Normaliser les valeurs de z pour les utiliser comme couleurs
norm = plt.Normalize(min(points_z), max(points_z))
colors = plt.cm.Reds(norm(points_z))

# Visualisation des points avec un gradient de rouge en fonction de z
ax.scatter(points_x, points_y, points_z, c=colors, s=1)
ax.set_box_aspect([1, 1, taille[0]/550*maxz/maxx])  # Aspect ratio is 2:2:1 (X:Y:Z)
# Paramètres de la visualisation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Couches')
plt.show()

