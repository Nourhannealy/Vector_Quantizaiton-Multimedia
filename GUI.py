# File name: VQ_GUI.py
# About: Vector Quantizer for image compression with a Tkinter GUI
# Authors:
#    Daad Amar Osman 20230779
#    Nourhan Aly Zakaria 20230453


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import math
import pickle
import random # Used for codebook initialization

# --- Vector Quantizer Algorithm (Unchanged from original logic) ---

class VectorQuantizer:
    def __init__(self, block_height, block_width, num_codewords):
        self.block_height = block_height
        self.block_width = block_width
        self.num_codewords = num_codewords
        self.codebook = None
        self.original_shape = None
        self.padded_shape = None
    
    def pad_image(self, image):
        """Pad image to be divisible by block dimensions"""
        w, h = image.size
        self.original_shape = (h, w)
        pad_h = (self.block_height - (h % self.block_height)) % self.block_height
        pad_w = (self.block_width - (w % self.block_width)) % self.block_width
        padded = Image.new('L', (w + pad_w, h + pad_h), color=0)
        padded.paste(image, (0, 0))
        self.padded_shape = (h + pad_h, w + pad_w)
        return padded
    
    def split_into_vectors(self, image):
        #split image into block vectors
        w, h = image.size
        vectors = []
        pixels = list(image.getdata())
        for i in range(0, h, self.block_height):
            for j in range(0, w, self.block_width):
                vector = []
                for bi in range(self.block_height):
                    for bj in range(self.block_width):
                        pixel_idx = (i + bi) * w + (j + bj)
                        vector.append(pixels[pixel_idx])
                vectors.append(vector)
        
        return vectors
    
    #calculate the distance between two vectors
    def distance(self, v1, v2):
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
    
    #calculate mean of multiple vectors
    def vector_mean(self, vectors):
        if not vectors:
            return None
        
        vector_len = len(vectors[0])
        mean = []
        for i in range(vector_len):
            values = [v[i] for v in vectors]
            mean.append(sum(values) / len(values))
        
        return mean
    
    #find the closest codeword to the vector
    def find_closest_codeword(self, vector, codebook):
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, codeword in enumerate(codebook):
            distance = self.distance(vector, codeword)
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        
        return closest_idx
    
    #create codebook
    def create_codebook(self, vectors):
        # print(f"Creating codebook with {self.num_codewords} codewords...")
        
        # Initialize codebook with random vectors
        random.seed(42)
        
        # Handle case where num_codewords is greater than total vectors
        num_vectors = len(vectors)
        k = min(self.num_codewords, num_vectors)
        
        if k == 0:
            self.codebook = []
            return []
            
        indices = random.sample(range(num_vectors), k)
        codebook = [vectors[i] for i in indices]
        for iteration in range(5):
            
            # Find closest codeword for each vector
            assignments = []
            for vector in vectors:
                closest_idx = self.find_closest_codeword(vector, codebook)
                assignments.append(closest_idx)
            
            # Update codewords
            new_codebook = []
            for i in range(self.num_codewords):
                cluster_vectors = [vectors[j] for j in range(num_vectors)
                                 if assignments[j] == i]
                
                if cluster_vectors:
                    new_codeword = self.vector_mean(cluster_vectors)
                    new_codebook.append(new_codeword)
                else:
                    # Keep the old codeword if the cluster is empty
                    # This only happens if i >= k (if k < self.num_codewords) or if
                    # an existing cluster becomes empty, which is handled here
                    if i < len(codebook):
                         new_codebook.append(codebook[i])
                    else:
                         # If we initially took fewer than num_codewords, pad with a dummy vector
                         # This should ideally not happen if k=min(num_codewords, num_vectors)
                         new_codebook.append([0] * self.block_height * self.block_width)
            
            # If the initial codebook was smaller than requested, ensure it grows to the target size
            while len(new_codebook) < self.num_codewords:
                # Simple fallback: duplicate the last good codeword or use zero vector
                if new_codebook:
                    new_codebook.append(new_codebook[-1])
                else:
                    new_codebook.append([0] * self.block_height * self.block_width)


            # Only update codebook up to the target size
            codebook = new_codebook[:self.num_codewords]
        
        self.codebook = codebook
        return codebook
    
    def compress(self, image_path, output_dir='output'):
        """Compress image"""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            image = Image.open(image_path).convert('L')
        except Exception as e:
            print(f"Error: Could not read image: {e}")
            return False, 0
        
        original_size = os.path.getsize(image_path)
        
        # Pad image
        padded_image = self.pad_image(image)
        
        # Split into vectors
        vectors = self.split_into_vectors(padded_image)
        
        # Create codebook
        self.create_codebook(vectors)
        
        if not self.codebook:
             return False, 0

        # Encode: find closest codeword for each vector
        encoded_indices = []
        for vector in vectors:
            closest_idx = self.find_closest_codeword(vector, self.codebook)
            encoded_indices.append(closest_idx)
        
        # Save codebook
        codebook_path = os.path.join(output_dir, 'codebook.pkl')
        with open(codebook_path, 'wb') as f:
            pickle.dump({
                'codebook': self.codebook,
                'original_shape': self.original_shape,
                'padded_shape': self.padded_shape,
                'block_height': self.block_height,
                'block_width': self.block_width
            }, f)
        
        # Save encoded indices
        indices_path = os.path.join(output_dir, 'encoded_indices.pkl')
        with open(indices_path, 'wb') as f:
            pickle.dump(encoded_indices, f)
        
        # Calculate sizes and compression ratio
        bits_per_index = math.ceil(math.log2(self.num_codewords)) if self.num_codewords > 1 else 1
        # Calculate compressed size based on indices storage (in bits, converted to bytes)
        compressed_size_indices = (len(encoded_indices) * bits_per_index) / 8
        # Estimate Codebook size (approx 4 bytes/float * vector_dim * num_codewords)
        vector_dim = self.block_height * self.block_width
        compressed_size_codebook = self.num_codewords * vector_dim * 4 
        
        compressed_size = compressed_size_indices + compressed_size_codebook
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return True, compression_ratio, codebook_path, indices_path
    
    def decompress(self, codebook_path, indices_path, output_path):

        # Load codebook + metadata
        try:
            with open(codebook_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading codebook: {e}")
            return False

        codebook = data['codebook']
        original_h, original_w = data['original_shape']
        padded_h, padded_w = data['padded_shape']
        block_h = data['block_height']
        block_w = data['block_width']

        # Load encoded indices
        try:
            with open(indices_path, 'rb') as f:
                encoded_indices = pickle.load(f)
        except Exception as e:
            print(f"Error loading indices: {e}")
            return False

        # Prepare empty pixel list for padded reconstruction
        reconstructed_pixels = [0] * (padded_h * padded_w)

        index = 0
        for i in range(0, padded_h, block_h):
            for j in range(0, padded_w, block_w):
                if index >= len(encoded_indices):
                    # Should not happen if files are paired correctly
                    break 

                codeword = codebook[encoded_indices[index]]
                index += 1

                # Fill block pixels
                p = 0
                for bi in range(block_h):
                    for bj in range(block_w):
                        if p >= len(codeword): # Safety check
                            break
                        
                        y = i + bi
                        x = j + bj
                        # Scale pixel values to 0-255 range and ensure integer
                        pixel_value = int(max(0, min(255, codeword[p])))
                        reconstructed_pixels[y * padded_w + x] = pixel_value
                        p += 1

        # Convert list back to Pillow image
        reconstructed_image = Image.new('L', (padded_w, padded_h))
        reconstructed_image.putdata(reconstructed_pixels)

        # Crop padded regions
        reconstructed_image = reconstructed_image.crop((0, 0, original_w, original_h))

        # Save final output image
        reconstructed_image.save(output_path)
        
        return True, reconstructed_image

# --- GUI Implementation ---

class VQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vector Quantizer Image Compressor")
        self.geometry("1200x800")
        self.configure(bg="#f4f7f9")
        
        # State variables
        self.image_path = None
        self.compressed_codebook_path = None
        self.compressed_indices_path = None
        self.original_tk_img = None
        self.reconstructed_tk_img = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main Layout: 2 Columns
        main_frame = ttk.Frame(self, padding="15", style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure styles
        s = ttk.Style()
        s.configure('TButton', font=('Arial', 10), padding=6, background='#007bff', foreground='black')
        s.configure('TLabel', font=('Arial', 10), background='#f4f7f9', foreground='#333333')
        s.configure('TEntry', font=('Arial', 10), padding=4)
        s.configure('Main.TFrame', background='#f4f7f9')
        s.configure('Panel.TFrame', background='#ffffff', relief='flat', borderwidth=2)
        s.configure('Header.TLabel', font=('Arial', 14, 'bold'), foreground='#007bff')

        # Left Panel (Controls - 40%)
        control_panel = ttk.Frame(main_frame, padding="15", style='Panel.TFrame')
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), ipadx=5, ipady=5)
        
        # Right Panel (Images - 60%)
        image_panel = ttk.Frame(main_frame, style='Main.TFrame')
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Compression Controls (Left Panel Top) ---
        ttk.Label(control_panel, text="1. Image Compression", style='Header.TLabel').pack(fill=tk.X, pady=(0, 10))

        # Input Path
        ttk.Label(control_panel, text="Original Image Path:").pack(fill=tk.X, pady=(5, 0))
        self.image_path_var = tk.StringVar()
        self.image_path_entry = ttk.Entry(control_panel, textvariable=self.image_path_var, width=50, state='readonly')
        self.image_path_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(control_panel, text="Select Image (PNG/JPG)", command=self.load_image_for_compression).pack(fill=tk.X, pady=(0, 10))

        # Parameters
        param_frame = ttk.Frame(control_panel, style='Panel.TFrame', padding="10")
        param_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(param_frame, text="Block Height (H):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.block_h_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.block_h_var, width=10).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(param_frame, text="Block Width (W):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.block_w_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.block_w_var, width=10).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(param_frame, text="Codewords (K):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.codewords_var = tk.StringVar(value="256")
        ttk.Entry(param_frame, textvariable=self.codewords_var, width=10).grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        
        param_frame.grid_columnconfigure(1, weight=1)

        # Compression Button & Status
        ttk.Button(control_panel, text="Run Compression (Generate PKL files)", command=self.start_compression, style='TButton').pack(fill=tk.X, pady=(5, 15))
        
        self.compression_status_var = tk.StringVar(value="Status: Ready to compress.")
        ttk.Label(control_panel, textvariable=self.compression_status_var, wraplength=300).pack(fill=tk.X)


        # Separator
        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=15)
        
        # --- Decompression Controls (Left Panel Bottom) ---
        ttk.Label(control_panel, text="2. Image Decompression", style='Header.TLabel').pack(fill=tk.X, pady=(0, 10))

        # Input Compressed Files
        ttk.Label(control_panel, text="Compressed Files Path:").pack(fill=tk.X, pady=(5, 0))
        self.codebook_path_var = tk.StringVar()
        ttk.Entry(control_panel, textvariable=self.codebook_path_var, width=50, state='readonly').pack(fill=tk.X, pady=(0, 2))
        self.indices_path_var = tk.StringVar()
        ttk.Entry(control_panel, textvariable=self.indices_path_var, width=50, state='readonly').pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(control_panel, text="Load Codebook & Indices", command=self.load_compressed_files).pack(fill=tk.X, pady=(0, 10))

        # Output Name
        ttk.Label(control_panel, text="Output Reconstructed Filename:").pack(fill=tk.X, pady=(5, 0))
        self.output_filename_var = tk.StringVar(value="reconstructed.png")
        ttk.Entry(control_panel, textvariable=self.output_filename_var, width=50).pack(fill=tk.X, pady=(0, 10))

        # Decompression Button
        ttk.Button(control_panel, text="Run Decompression (Generate PNG)", command=self.start_decompression, style='TButton').pack(fill=tk.X, pady=(5, 10))
        
        self.decompression_status_var = tk.StringVar(value="Status: Ready to decompress.")
        ttk.Label(control_panel, textvariable=self.decompression_status_var, wraplength=300).pack(fill=tk.X)
        
        
        # --- Image Display (Right Panel) ---
        
        # Image Containers
        image_container = ttk.Frame(image_panel, style='Main.TFrame')
        image_container.pack(fill=tk.BOTH, expand=True)

        # Original Image
        original_frame = ttk.Frame(image_container, padding="10", style='Panel.TFrame')
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        ttk.Label(original_frame, text="Original Image", style='Header.TLabel').pack(pady=(0, 10))
        self.original_label = ttk.Label(original_frame, text="Load an image...", anchor=tk.CENTER)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # Reconstructed Image
        reconstructed_frame = ttk.Frame(image_container, padding="10", style='Panel.TFrame')
        reconstructed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        ttk.Label(reconstructed_frame, text="Reconstructed Image", style='Header.TLabel').pack(pady=(0, 10))
        self.reconstructed_label = ttk.Label(reconstructed_frame, text="Decompress to view...", anchor=tk.CENTER)
        self.reconstructed_label.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty images
        self.display_image(None, self.original_label)
        self.display_image(None, self.reconstructed_label)
        
        # Binding resize event to image panel
        image_panel.bind("<Configure>", self.on_resize)
        self.current_original_image = None
        self.current_reconstructed_image = None

    def on_resize(self, event):
        # Redraw images on window resize
        self.display_image(self.current_original_image, self.original_label)
        self.display_image(self.current_reconstructed_image, self.reconstructed_label)

    def display_image(self, pil_image, label):
        # Clear previous image reference
        label.image = None 
        
        if pil_image is None:
            # Display a placeholder text
            label.config(image='', text="[Image Placeholder]")
            return

        # Get the size of the label frame
        w = label.winfo_width() - 20 # Account for padding
        h = label.winfo_height() - 20

        if w <= 0 or h <= 0:
            # Cannot resize if size is 0
            label.config(image='', text="[Image Loaded, resize window to view]")
            return

        # Resize image to fit label area while maintaining aspect ratio
        img_w, img_h = pil_image.size
        ratio = min(w / img_w, h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        if new_w > 0 and new_h > 0:
            resized_img = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_img)
            label.config(image=tk_img, text="")
            label.image = tk_img # Keep a reference
        else:
             label.config(image='', text="[Image Loaded, resize window to view]")


    def load_image_for_compression(self):
        f_path = filedialog.askopenfilename(
            defaultextension=".png",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if f_path:
            self.image_path = f_path
            self.image_path_var.set(os.path.basename(f_path))
            self.compression_status_var.set(f"Image selected: {os.path.basename(f_path)}. Ready.")
            
            try:
                img = Image.open(f_path).convert('L')
                self.current_original_image = img
                self.display_image(img, self.original_label)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.image_path = None
                self.image_path_var.set("")
                self.current_original_image = None
                self.display_image(None, self.original_label)

    def start_compression(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        try:
            block_h = int(self.block_h_var.get())
            block_w = int(self.block_w_var.get())
            num_codewords = int(self.codewords_var.get())
            
            if block_h <= 0 or block_w <= 0 or num_codewords <= 0:
                messagebox.showerror("Error", "Block dimensions and codeword count must be positive integers.")
                return

        except ValueError:
            messagebox.showerror("Error", "Invalid input for parameters. Please use integers.")
            return

        self.compression_status_var.set("Status: Running compression...")
        self.update_idletasks()

        try:
            vq = VectorQuantizer(block_h, block_w, num_codewords)
            output_dir = 'output' # Default output directory
            success, ratio, codebook_path, indices_path = vq.compress(self.image_path, output_dir)
            
            if success:
                # Update decompression paths automatically
                self.compressed_codebook_path = codebook_path
                self.compressed_indices_path = indices_path
                self.codebook_path_var.set(os.path.basename(codebook_path))
                self.indices_path_var.set(os.path.basename(indices_path))
                
                self.compression_status_var.set(
                    f"Compression SUCCESS! Ratio: {ratio:.2f}:1.\nFiles saved to '{output_dir}/'."
                )
                self.decompression_status_var.set("Ready to decompress using newly created files.")
            else:
                self.compression_status_var.set("Compression FAILED. Check console for details.")

        except Exception as e:
            messagebox.showerror("Compression Error", f"An unexpected error occurred during compression: {e}")
            self.compression_status_var.set("Compression FAILED due to unexpected error.")


    def load_compressed_files(self):
        codebook_path = filedialog.askopenfilename(
            title="Select Codebook File (codebook.pkl)",
            filetypes=[("Pickle files", "*.pkl")]
        )
        if not codebook_path:
            return

        indices_path = filedialog.askopenfilename(
            title="Select Indices File (encoded_indices.pkl)",
            filetypes=[("Pickle files", "*.pkl")]
        )
        if not indices_path:
            return
            
        self.compressed_codebook_path = codebook_path
        self.compressed_indices_path = indices_path
        self.codebook_path_var.set(os.path.basename(codebook_path))
        self.indices_path_var.set(os.path.basename(indices_path))
        self.decompression_status_var.set("Compressed files loaded. Ready for decompression.")


    def start_decompression(self):
        if not self.compressed_codebook_path or not self.compressed_indices_path:
            messagebox.showwarning("Warning", "Please load the codebook and indices files first.")
            return

        output_filename = self.output_filename_var.get()
        if not output_filename:
            messagebox.showerror("Error", "Please provide an output filename.")
            return

        # Ensure output directory exists before creating the full path
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        self.decompression_status_var.set("Status: Running decompression...")
        self.update_idletasks()

        try:
            # VQ initializer is dummy here as its values will be overwritten by loaded metadata
            vq = VectorQuantizer(1, 1, 1) 
            success, reconstructed_image = vq.decompress(
                self.compressed_codebook_path, 
                self.compressed_indices_path, 
                output_path
            )
            
            if success:
                self.current_reconstructed_image = reconstructed_image
                self.display_image(reconstructed_image, self.reconstructed_label)
                
                self.decompression_status_var.set(
                    f"Decompression SUCCESS! Image saved to '{output_path}'."
                )
            else:
                self.decompression_status_var.set("Decompression FAILED. Check console for file errors.")

        except Exception as e:
            messagebox.showerror("Decompression Error", f"An unexpected error occurred during decompression: {e}")
            self.decompression_status_var.set("Decompression FAILED due to unexpected error.")


if __name__ == "__main__":
    app = VQApp()
    app.mainloop()