import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tifffile
from use_model import *



def apply_function(large_test_images):
    # Example function: invert the image
    return run_model(large_test_images,save =False)
    

def update_image_display():
    try:
        current_slice = original_image.image_array[original_image.current_index]

        # Update the original image display
        pil_image = Image.fromarray(current_slice.astype(np.uint8))
        original_image.display_image = ImageTk.PhotoImage(pil_image)
        original_label.config(image=original_image.display_image)
        original_label.image = original_image.display_image

        # Update the processed image display
        if hasattr(processed_image, 'image_array'):
            processed_slice = processed_image.image_array[original_image.current_index]
            pil_processed_image = Image.fromarray(processed_slice.astype(np.uint8))
            processed_image.display_image = ImageTk.PhotoImage(pil_processed_image)
            processed_label.config(image=processed_image.display_image)
            processed_label.image = processed_image.display_image
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update image display: {e}")

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif;*.tiff")])
    if not file_path:
        return

    try:
        image_array = tifffile.imread(file_path)
        original_image.image_array = image_array

        # Normalize and handle 3D grayscale images
        if len(image_array.shape) == 3 and image_array.dtype != np.uint8:
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)

        original_image.image_array = image_array
        original_image.current_index = 0
        original_image.is_3d = len(image_array.shape) == 3

        # Update scrollbar range
        scrollbar.config(to=image_array.shape[0] - 1 if original_image.is_3d else 0)
        scrollbar.set(0)
        update_image_display()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")

def process_image():
    if not hasattr(original_image, 'image_array'):
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    try:
        processed_image.image_array = apply_function(original_image.image_array)
        update_image_display()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")

def export_image():
    if not hasattr(processed_image, 'image_array'):
        messagebox.showwarning("Warning", "No processed image to export.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif;*.tiff")])
    if not file_path:
        return

    try:
        tifffile.imwrite(file_path, processed_image.image_array.astype(np.uint8))
        messagebox.showinfo("Success", "Processed image exported successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export image: {e}")

def on_scroll(value):
    if not original_image.is_3d:
        return

    original_image.current_index = int(value)
    update_image_display()

# Initialize main Tkinter window
root = tk.Tk()
root.title("TIFF Image Visualizer")

# Frames for layout
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X)

image_frame = tk.Frame(root)
image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

scroll_frame = tk.Frame(root)
scroll_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Buttons for controls
load_button = tk.Button(control_frame, text="Load TIFF", command=load_image)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

process_button = tk.Button(control_frame, text="Process Image", command=process_image)
process_button.pack(side=tk.LEFT, padx=5, pady=5)

export_button = tk.Button(control_frame, text="Export Processed TIFF", command=export_image)
export_button.pack(side=tk.LEFT, padx=5, pady=5)

# Scrollbar for navigation
scrollbar = tk.Scale(scroll_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_scroll)
scrollbar.pack(fill=tk.X, padx=5, pady=5)

# Labels for image display
original_label = tk.Label(image_frame, text="Original Image", bg="gray", width=50, height=25)
original_label.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.BOTH)

processed_label = tk.Label(image_frame, text="Processed Image", bg="gray", width=50, height=25)
processed_label.pack(side=tk.RIGHT, padx=5, pady=5, expand=True, fill=tk.BOTH)

# Containers for image data
original_image = tk.Label()
processed_image = tk.Label()
original_image.is_3d = False
original_image.current_index = 0

# Run the Tkinter event loop

root.mainloop()
