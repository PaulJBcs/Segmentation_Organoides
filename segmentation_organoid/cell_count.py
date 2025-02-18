from __future__ import division
from PIL import Image
import cv2
import numpy as np

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


def count_area(image):
    '''entrée : mask : image binaire'''
    ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    #ret,thresh1 = cv2.threshold(blur,168,255,cv2.THRESH_BINARY)
    thresh1_3ch = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(thresh1_3ch, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((4,4), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    li = []
    n = len(contours)
    for i in range(n):
        area = cv2.contourArea(contours[i])
        M = cv2.moments(contours[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        li.append((area,(cx,cy)))
    return n,li

m = 0
dic = {}

def main(path,save = True):
    global m
    global dic
    images_list = tiff_to_list(path)
    nb_couche = len(images_list)
    def iter_count_3D(i_c,li1,li2,li3,val = 1):
        global dic
        global m
        '''une iteration entre deux couches (1 en haut et 2 en bas) 3 encore en dessous'''
        for cell in li1:
            min = np.inf
            for cell2 in li2:
                distance = np.linalg.norm(np.array(cell[1])-np.array(cell2[1]))
                if distance < min:
                    min = distance
                    cell2_min = cell2
            if li2 != [] and min < val*np.sqrt(cell2_min[0])/np.pi:
                try:
                    id = dic[(cell2_min,i_c-1)]
                except:
                    print(f"erreur : {cell2_min,i_c-1}", cell,i_c)
                    print(((70.5, (50, 507)), 209) in dic)
                    return
                dic[(cell,i_c)] = id
            else:
                min = np.inf
                for cell2 in li3:
                    distance = np.linalg.norm(np.array(cell[1])-np.array(cell2[1]))
                    if distance < min:
                        min = distance
                        cell2_min = cell2
                if li2 != [] and min < val*np.sqrt(cell2_min[0])/np.pi:
                    id = dic[(cell2_min,i_c-2)]
                    dic[(cell,i_c)] = id
                    dic[(cell,i_c-1)] = id
                else :
                    dic[(cell,i_c)] = m
                    m+=1
                    
                

    mask0 = pil_to_cv2(images_list[0])
    prev_0,prev_li = count_area(mask0)
    prev_prev_li = prev_li 
    for cell in prev_li:
        dic[cell,0] = m
        m+=1

    for i_c in range(nb_couche):
        new_mask = pil_to_cv2(images_list[i_c])
        new_n,new_li = count_area(new_mask)
        iter_count_3D(i_c,new_li,prev_li,prev_prev_li)
        prev_prev_li= prev_li
        prev_li = new_li
        prev_n = new_n
   
    dic_final = {}
    for cell in dic:
        cell_clean = (cell[0][0],cell[0][1],cell[1])
        if dic[cell] in dic_final:
            dic_final[dic[cell]].append(cell_clean)
        else:
            dic_final[dic[cell]] = [cell_clean]
    res = []
    for key in dic_final:
        volume = sum([x[0] for x in dic_final[key]])
        posx,posy,posz = sum([x[0]*x[1][0] for x in dic_final[key]])/volume,sum([x[0]*x[1][1] for x in dic_final[key]])/volume, sum([x[0]*x[2] for x in dic_final[key]])/volume
        #print(f"cellule {key} : volume : {volume}, barycentre : ({posx},{posy},{posz})")
        if len(dic_final[key]) == 1:
            # print(f"cellule {key} : volume : {volume}, barycentre : ({posx},{posy},{posz})")
            pass
        else :
            res.append((11*11*4.5*volume,(posx,posy,posz)))
    return res

if __name__ == "__main__":
    tiff_path = input("entrez le fichier au format TIFF :")
    li = main(tiff_path,save = False) 
    volume_tot = 0 
    for i,cell  in enumerate(li):
        volume_tot += cell[0]
        #print(f"cellule {i} : volume : {cell[0]} µm³, barycentre : {cell[1]}")
    print(f"nombre de cellules : {len(li)} , volume totale  des cellules : {volume_tot} µm³, cellule moyenne : {volume_tot/len(li)} µm³")