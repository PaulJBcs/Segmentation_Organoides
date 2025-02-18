import cv2
from skimage.transform import resize
import json
from scipy.spatial.distance import dice
from PIL import Image, ImageEnhance
import os
import tifffile
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt
import torch
from transformers import SamModel, SamConfig, SamProcessor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("Starting ...")
# from segment_anything import SamModel, SamConfig, SamProcessor
# from cell_count import *

print("Loading the model")
# Load the model configuration
model_config = SamConfig.from_pretrained("flaviagiammarino/medsam-vit-base")
processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create an instance of the model architecture with the loaded configuration
model = SamModel(config=model_config)


def superpose_images_with_translucent_overlay(image1, image2, output_path, overlay_color=(255, 255, 0), transparency=128):
    """
    Superpose deux images en appliquant une teinte à la deuxième image et en la rendant translucide.

    Args:
        image1_path (str): Chemin de l'image de fond.
        image2_path (str): Chemin de l'image à superposer.
        output_path (str): Chemin pour enregistrer l'image résultante.
        overlay_color (tuple): Couleur de la teinte appliquée à la deuxième image (par défaut: jaune).
        transparency (int): Niveau de transparence de la deuxième image (0-255).
    """

    # Créer une image colorée où le blanc devient jaune et le noir reste noir
    image2_colored_array = np.zeros(
        (image2.shape[0], image2.shape[1], 4), dtype=np.uint8)
    image2_colored_array[image2 == 1] = overlay_color + (255,)
    image2_colored_array[image2 == 0] = (0, 0, 0, 255)

    # Convertir le tableau NumPy en image PIL
    image2_colored = Image.fromarray(image2_colored_array, mode="RGBA")

    # Appliquer la transparence à image2
    image2_colored.putalpha(transparency)

    image1 = Image.fromarray(image1).convert(
        "RGBA")  # Image de fon  # Image à superposer
    # Superposer les deux images
    result = Image.alpha_composite(image1, image2_colored)

    # Enregistrer l'image résultante
    return np.array(result).astype(np.uint8)


def inference(single_patch, threshold):
    """inference du model sur un patch"""
    inputs = processor(single_patch, input_boxes=[
                       [[0, 0, 255, 255]]], return_tensors="pt")

    # Move the input tensor to the GPU if it's not already there
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # apply sigmoid
    single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
    single_patch_prediction = (single_patch_prob > threshold).astype(np.uint8)

    return single_patch_prob, single_patch_prediction


def load_image(all_test_images, idx, all_test_masks=None):
    large_test_image = all_test_images[idx]
    # Step=256 for 256 patches means no overlap
    patches = patchify(large_test_image, (256, 256), step=256)
    i, j = 0, 1
    single_patch = patches[i, j]
    single_patch = np.stack((single_patch,) * 3, axis=-1)
    single_patch = Image.fromarray(single_patch.astype(np.uint8))
    single_patch_prob, single_patch_prediction = inference(single_patch, 0.7)
    if all_test_masks == None:
        return single_patch, single_patch_prob, single_patch_prediction

    large_test_mask = all_test_masks[idx]

    # Step=256 for 256 patches means no overlap
    patches_mask = patchify(large_test_mask, (256, 256), step=256)

    single_patch_mask = patches_mask[i, j]
    single_patch_mask = np.stack((single_patch_mask,) * 3, axis=-1)
    single_patch_mask = Image.fromarray(single_patch_mask.astype(np.uint8))

    return single_patch, single_patch_mask, single_patch_prob, single_patch_prediction


def plot_result(single_patch, single_patch_mask, single_patch_prob, single_patch_prediction, path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the first image on the left
    # Assuming the first image is grayscale
    axes[0].imshow(np.array(single_patch), cmap='gray')
    axes[0].set_title("Image")

    # Assuming the second image is grayscale
    axes[1].imshow(np.array(single_patch_mask))
    axes[1].set_title("Mask")

    # Plot the second image on the right
    axes[2].imshow(single_patch_prob)  # Assuming the second image is grayscale
    axes[2].set_title("Probability Map")

    # Plot the second image on the right
    # Assuming the second image is grayscale
    axes[3].imshow(single_patch_prediction, cmap='gray')
    axes[3].set_title("Prediction")

    # Display the images side by side
    plt.savefig(path)


def dice_coefficient(image1, image2):
    # Convertir les images PIL en tableaux NumPy binaires
    array1 = np.array(image1).astype(bool).ravel()
    array2 = np.array(image2).astype(bool).ravel()
    intersection = np.sum(array1 & array2)
    sum_pixels = np.sum(array1) + np.sum(array2)
    # Gérer le cas où les deux masques sont vides
    if sum_pixels == 0:
        return [1.0, 1.0]  # Les deux masques sont identiques et vides

    # Calcul du coefficient de Dice

    dice = 2 * intersection / sum_pixels
    IoU = intersection/(sum_pixels-intersection)
    return [dice, IoU]


def run_model(large_test_images, save=False):
    if large_test_images.all == None:
        path_images = input(
            "Entrez le chemin d'accès vers le fichier à analiser (sous format .tif) :")
        large_test_images = tifffile.imread(path_images)
    elif type(large_test_images) == str:
        large_test_images = tifffile.imread(large_test_images)
    print("analizing images with the model")
    warnings.filterwarnings("ignore", category=UserWarning)
    import torch
    # model.load_state_dict(torch.load(f"model_checkpoint_12.pth"), map_location=torch.device('cpu'), strict=False)
    model.load_state_dict(torch.load(
        "model_checkpoint_12.pth", map_location=torch.device('cpu')), strict=False)

    # Ignore warning messages
    model.to(device)
    L = []
    L2 = []
    # n = len(large_test_images)
    n = 20

    for i in range(n):
        print(i, "/", n, flush=True)
        single_patch, single_patch_prob, single_patch_prediction = load_image(
            large_test_images, i)
        pred = single_patch_prediction.astype(np.uint8)
        img = np.array(single_patch).astype(np.uint8)
        L.append(pred)
        L2.append(superpose_images_with_translucent_overlay(
            img, pred, f"results/result_{i}.png"))
    tifffile.imwrite("prediction.tif", L)
    tifffile.imwrite("images_superpose.tif", L2)
    if save:
        save_tiff("pred", "prediction.tif")
        save_tiff("result", "images_superpose.tif")
    return tifffile.imread("images_superpose.tif")


def save_tiff(name, path):
    images = tifffile.imread(path)
    for i, image in enumerate(images):
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Normaliser les valeurs entre 0 et 255
            image = (255 * (image - np.min(image)) /
                     (np.max(image) - np.min(image))).astype(np.uint8)
        pil_image = Image.fromarray(image)
        output_path = f"results/{name}_{i + 1}.png"
        pil_image.save(output_path, format="PNG")


if __name__ == "__main__":
    run_model(large_test_images=None, save=True)
    print("Done")


# with open("list_dice.json", "r") as f:
#    L = json.load(f)
