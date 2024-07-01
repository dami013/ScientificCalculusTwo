import numpy as np
import customtkinter
from tkinter import filedialog, messagebox
from customtkinter import CTkImage
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from DCT_lib import dct2_library, idct2_library

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("DCT2 Image Processor")
        self.geometry("800x600")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = customtkinter.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Left column
        self.left_frame = customtkinter.CTkFrame(self.main_frame)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.file_button = customtkinter.CTkButton(self.left_frame, text="Select BMP Image", command=self.open_file)
        self.file_button.pack(pady=(0, 20), padx=10, fill="x")

        self.img_preview = customtkinter.CTkLabel(self.left_frame, text="No image selected", width=400, height=400)
        self.img_preview.pack(pady=10, padx=10)

        # Right column
        self.right_frame = customtkinter.CTkFrame(self.main_frame)
        self.right_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.params_frame = customtkinter.CTkFrame(self.right_frame)
        self.params_frame.pack(pady=10, padx=10, fill="x")

        self.F_label = customtkinter.CTkLabel(self.params_frame, text="F (macro-block size):")
        self.F_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.F_entry = customtkinter.CTkEntry(self.params_frame, width=100)
        self.F_entry.grid(row=0, column=1, padx=5, pady=5)

        self.d_label = customtkinter.CTkLabel(self.params_frame, text="d (threshold):")
        self.d_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.d_entry = customtkinter.CTkEntry(self.params_frame, width=100)
        self.d_entry.grid(row=1, column=1, padx=5, pady=5)

        self.apply_button = customtkinter.CTkButton(self.right_frame, text="Apply DCT2", command=self.apply_dct)
        self.apply_button.pack(pady=10, padx=10, fill="x")

        self.reset_button = customtkinter.CTkButton(self.right_frame, text="Reset", command=self.reset)
        self.reset_button.pack(pady=10, padx=10, fill="x")

        self.save_button = customtkinter.CTkButton(self.right_frame, text="Save Processed Image", command=self.save_image)
        self.save_button.pack(pady=10, padx=10, fill="x")
        self.save_button.configure(state="disabled")

        # Status bar
        self.status_bar = customtkinter.CTkLabel(self.main_frame, text="Ready", anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 0))

        self.img_array = None
        self.processed_image = None

    def apply_dct2(self, matrix, F, d):
        n, m = matrix.shape
        dct_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(0, n - n % F, F):
            for j in range(0, m - m % F, F):
                block = matrix[i:i + F, j:j + F]
                block_dct = dct2_library(block)
                for k in range(F):
                    for l in range(F):
                        if k + l >= d:
                            block_dct[k, l] = 0
                block_idct = idct2_library(block_dct)
                # Arrotondare, limitare e assegnare i valori
                block_idct = np.round(block_idct)
                block_idct[block_idct < 0] = 0
                block_idct[block_idct > 255] = 255
                dct_matrix[i:i + F, j:j + F] = block_idct
        return dct_matrix

    def apply_dct(self):
        if self.img_array is None:
            messagebox.showerror("Error", "No image selected")
            return

        try:
            F = int(self.F_entry.get())
            d = int(self.d_entry.get())
        except ValueError:
            messagebox.showerror("Error", "F and d must be integer values")
            return

        if not (0 <= d <= 2 * F - 2):
            messagebox.showerror("Error", "d must be between 0 and 2F-2")
            return

        modified_image = self.apply_dct2(self.img_array, F, d)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(self.img_array, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Modified Image')
        plt.imshow(modified_image, cmap='gray')
        plt.axis('off')
        plt.show()

        self.processed_image = modified_image
        self.save_button.configure(state="normal")

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            try:
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                self.img_array = np.array(img)
                
                # Resize image for preview
                img.thumbnail((200, 200))  # Increase the size to 400x400 for preview
                ctk_image = CTkImage(light_image=img, dark_image=img, size=(400, 400))
                self.img_preview.configure(image=ctk_image, text="")
                self.img_preview.image = ctk_image
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")

    def reset(self):
        self.F_entry.delete(0, 'end')
        self.d_entry.delete(0, 'end')
        self.img_array = None
        self.processed_image = None
        
        # Recreate the img_preview widget
        self.img_preview.destroy()
        self.img_preview = customtkinter.CTkLabel(self.left_frame, text="No image selected", width=400, height=400)
        self.img_preview.pack(pady=10, padx=10)
        
        self.save_button.configure(state="disabled")

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP files", "*.bmp")])
            if file_path:
                Image.fromarray(self.processed_image.astype(np.uint8)).save(file_path)
                messagebox.showinfo("Success", "Image saved successfully!")

if __name__ == '__main__':
    app = App()
    app.mainloop()
