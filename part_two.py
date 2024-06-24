import numpy as np
import tkinter as tk
import customtkinter
from tkinter import filedialog, messagebox
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from PIL import Image
from DCT_lib import dct2_library


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("DCT2 Image Processor")
        self.geometry("400x200")
        customtkinter.set_appearance_mode("night")

        self.frame = customtkinter.CTkFrame(master=self, width=400, height=400)
        self.frame.pack(pady=10)

        self.file_button = customtkinter.CTkButton(self.frame, text="Select BMP Image", command=self.open_file)
        self.file_button.grid(row=0, column=0, padx=5, pady=5)

        self.img_label = customtkinter.CTkLabel(self.frame, text="No file selected")
        self.img_label.grid(row=0, column=1, padx=5, pady=5)

        self.F_label = customtkinter.CTkLabel(self.frame, text="Enter F (macro-block size):")
        self.F_label.grid(row=1, column=0, padx=5, pady=5)

        self.F_entry = customtkinter.CTkEntry(self.frame)
        self.F_entry.grid(row=1, column=1, padx=5, pady=5)

        self.d_label = customtkinter.CTkLabel(self.frame, text="Enter d (threshold):")
        self.d_label.grid(row=2, column=0, padx=5, pady=5)

        self.d_entry = customtkinter.CTkEntry(self.frame)
        self.d_entry.grid(row=2, column=1, padx=5, pady=5)

        self.apply_button = customtkinter.CTkButton(self.frame, text="Apply DCT2", command=self.apply_dct)
        self.apply_button.grid(row=3, columnspan=2, pady=10)

    def apply_dct2(self, matrix, F, d):
        h, w = matrix.shape
        dct_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(0, h, F):
            for j in range(0, w, F):
                block = matrix[i:i + F, j:j + F]
                block_dct = dct2_library(block)
                block_dct[d:, d:] = 0
                block_idct = dct2_library(block_dct)
                dct_matrix[i:i + F, j:j + F] = block_idct
        return dct_matrix

    def apply_dct(self):
        F = int(self.F_entry.get())
        d = int(self.d_entry.get())
        img_array = self.img_label.img_array

        if img_array is None:
            messagebox.showerror("Error", "No image selected")
            return

        if not (0 <= d <= 2 * F - 2):
            messagebox.showerror("Error", "d must be between 0 and 2F-2")
            return

        result = self.apply_dct2(img_array, F, d)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_array, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("DCT2 Applied")
        plt.imshow(result, cmap='gray')
        plt.show()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            try:
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)
                self.img_label.configure(text=f"Selected file: {file_path}")
                self.img_label.img_array = img_array  # Store image array in the label widget
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")


if __name__ == '__main__':
    app = App()
    app.mainloop()
