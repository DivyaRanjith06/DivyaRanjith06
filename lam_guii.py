import cv2
import numpy as np
import ezdxf
from shapely.geometry import Polygon
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# Function to smooth and round contours
def offset_and_smooth_contour(contour, offset_distance=5.0, smoothing_factor=0.02, rounding=True):
    poly = Polygon(contour.reshape(-1, 2))
    join_style = 2 if rounding else 1
    offset_poly = poly.buffer(offset_distance, join_style=join_style)

    if not offset_poly.is_empty and offset_poly.exterior:
        smoothed_contour = np.array(offset_poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)
        contour_length = cv2.arcLength(smoothed_contour, True)
        epsilon = 0.01 * contour_length
        smoothed_contour = cv2.approxPolyDP(smoothed_contour, epsilon, True)
        return smoothed_contour
    else:
        return contour

# Function to detect contours and handle inner/outer ring shapes
def detect_and_smooth_contours(image, min_contour_area=100):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    _, thresh = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY_INV)
    
    edges = cv2.Canny(thresh, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_contour_area and hierarchy[0][i][3] == -1:  # Only include outer contours
            filtered_contours.append(cnt)

    smoothed_contours = [offset_and_smooth_contour(cnt) for cnt in filtered_contours]
    return smoothed_contours

# Function to compactly nest contours on a canvas
def compact_nest_contours(contours, canvas_size=(1000, 1000)):
    global nested_contours
    nested_image = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    nested_contours = []
    current_x, current_y, row_height = 10, 10, 0
    padding = 1

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if current_x + w + padding > canvas_size[1]:
            current_x = 10
            current_y += row_height + padding
            row_height = 0

        contour_shifted = contour - contour.min(axis=0)
        translation_matrix = np.array([[1, 0, current_x], [0, 1, current_y]], dtype=np.float32)
        shifted_contour = cv2.transform(contour_shifted, translation_matrix)
        
        cv2.drawContours(nested_image, [shifted_contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
        
        nested_contours.append(shifted_contour)

        current_x += w + padding
        row_height = max(row_height, h)

    return nested_image

# Function to export contours to a DXF file with 1:1 scaling in meters
def export_to_dxf(filename, scale_factor=1.0):
    if nested_contours:
        doc = ezdxf.new()
        msp = doc.modelspace()
        for contour in nested_contours:
            scaled_contour = contour.reshape(-1, 2) * scale_factor
            msp.add_lwpolyline(points=scaled_contour.tolist(), close=True)
        doc.saveas(filename)
    else:
        messagebox.showerror("Error", "No nested contours available for export.")

# Function to save nested image as JPEG
def save_jpeg():
    if 'nested_image' in globals():
        filename = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        if filename:
            cv2.imwrite(filename, nested_image)
            messagebox.showinfo("Save Complete", f"Nested image saved as JPEG to {filename}")
    else:
        messagebox.showerror("Error", "No nested image available to save.")

# Function to capture image
def capture_image():
    global frame, all_contours
    ret, frame = cam.read()
    if ret:
        contours = detect_and_smooth_contours(frame)
        all_contours.extend(contours)
        update_preview(frame)
        messagebox.showinfo("Image Captured", f"Captured frame with {len(contours)} contours.")
    else:
        messagebox.showerror("Error", "Failed to capture frame")

# Function to apply offset and round contours
def apply_offset_and_round():
    global processed_contours, nested_image
    offset_distance = simpledialog.askfloat("Offset Distance", "Enter offset distance (in pixels):", minvalue=0.0)
    smoothing_factor = simpledialog.askfloat("Smoothing Factor", "Enter smoothing factor (0 to 1):", minvalue=0.0, maxvalue=1.0)
    if offset_distance is not None and smoothing_factor is not None:
        processed_contours = [offset_and_smooth_contour(cnt, offset_distance, smoothing_factor, rounding=True) for cnt in all_contours]
        nested_image = compact_nest_contours(processed_contours)
        update_preview(nested_image)
        messagebox.showinfo("Processing Complete", "Contours offset and smoothed successfully.")

# Function to save DXF file
def save_dxf():
    if processed_contours:
        filename = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF files", "*.dxf")])
        
        # Replace this with your calculated pixel-to-meter conversion factor
        pixel_to_meter_ratio = 0.0007 
        
        if filename:
            export_to_dxf(filename, scale_factor=pixel_to_meter_ratio)
            messagebox.showinfo("Export Complete", f"Contours exported to {filename}")
    else:
        messagebox.showerror("Error", "No processed contours available for export.")

# Function to update the image preview
def update_preview(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    preview_label.config(image=img_tk)
    preview_label.image = img_tk

# Function to continuously capture frames for preview
def show_preview():
    global frame
    ret, frame = cam.read()
    if ret:
        update_preview(frame)
    root.after(10, show_preview)

# Main Application Window
cam = cv2.VideoCapture(1)
all_contours = []
processed_contours = []
nested_contours = []

root = tk.Tk()
root.title("Contour Offset and Export Tool")

capture_button = tk.Button(root, text="Capture Image", command=capture_image)
capture_button.pack(pady=10)

offset_button = tk.Button(root, text="Offset and Smooth Contours", command=apply_offset_and_round)
offset_button.pack(pady=10)

save_dxf_button = tk.Button(root, text="Save to DXF", command=save_dxf)
save_dxf_button.pack(pady=10)

save_jpeg_button = tk.Button(root, text="Save as JPEG", command=save_jpeg)
save_jpeg_button.pack(pady=10)

preview_label = tk.Label(root)
preview_label.pack(pady=10)

show_preview()

root.mainloop()
cam.release()
cv2.destroyAllWindows()
