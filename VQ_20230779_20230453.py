#File name: VQ_20230779_20230453.py
#About: Vector Quantizer for image compression
#Authors: 
#   Daad Amar Osman 20230779
#   Norhan Aly Zakaria 20230453


import pickle
import os
import math
from PIL import Image 
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
        print(f"Creating codebook with {self.num_codewords} codewords...")
        
        # Initialize codebook with random vectors
        import random
        random.seed(42)
        indices = random.sample(range(len(vectors)), self.num_codewords)
        codebook = [vectors[i] for i in indices]
        
        # Iterative refinement
        for iteration in range(5):
            print(f"  Iteration {iteration + 1}...")
            
            # Find closest codeword for each vector
            assignments = []
            for vector in vectors:
                closest_idx = self.find_closest_codeword(vector, codebook)
                assignments.append(closest_idx)
            
            # Update codewords
            new_codebook = []
            for i in range(self.num_codewords):
                cluster_vectors = [vectors[j] for j in range(len(vectors)) 
                                 if assignments[j] == i]
                
                if cluster_vectors:
                    new_codeword = self.vector_mean(cluster_vectors)
                    new_codebook.append(new_codeword)
                else:
                    new_codebook.append(codebook[i])
            
            codebook = new_codebook
        
        self.codebook = codebook
        return codebook
    
    def compress(self, image_path, output_dir='output'):
        """Compress image"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Read grayscale image
        try:
            image = Image.open(image_path).convert('L')
        except:
            print("Error: Could not read image")
            return False
        
        original_size = os.path.getsize(image_path)
        print(f"Original image size: {original_size} bytes")
        print(f"Original shape: {image.size[0]} x {image.size[1]}")
        
        # Pad image
        padded_image = self.pad_image(image)
        print(f"Padded shape: {self.padded_shape[1]} x {self.padded_shape[0]}")
        
        # Split into vectors
        print(f"\nSplitting image into blocks ({self.block_height}x{self.block_width})...")
        vectors = self.split_into_vectors(padded_image)
        print(f"Total vectors: {len(vectors)}")
        
        # Create codebook
        print("\nCreating codebook...")
        self.create_codebook(vectors)
        
        # Encode: find closest codeword for each vector
        print("\nEncoding vectors...")
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
        compressed_size = (len(encoded_indices) * bits_per_index) / 8
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"COMPRESSION COMPLETE")
        print(f"{'='*50}")
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size:.2f} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"Bits per index: {bits_per_index}")
        print(f"Codebook saved: {codebook_path}")
        print(f"Indices saved: {indices_path}")
        
        return True, compression_ratio
    
    def decompress(self, codebook_path, indices_path, output_path='reconstructed.png'):

        # Load codebook + metadata
        try:
            with open(codebook_path, 'rb') as f:
                data = pickle.load(f)
        except:
            print("Error: Could not load codebook file.")
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
        except:
            print("Error: Could not load indices file.")
            return False

        # Prepare empty pixel list for padded reconstruction
        reconstructed_pixels = [0] * (padded_h * padded_w)

        index = 0
        for i in range(0, padded_h, block_h):
            for j in range(0, padded_w, block_w):

                codeword = codebook[encoded_indices[index]]
                index += 1

                # Fill block pixels
                p = 0
                for bi in range(block_h):
                    for bj in range(block_w):
                        y = i + bi
                        x = j + bj
                        reconstructed_pixels[y * padded_w + x] = int(codeword[p])
                        p += 1

        # Convert list back to Pillow image
        reconstructed_image = Image.new('L', (padded_w, padded_h))
        reconstructed_image.putdata(reconstructed_pixels)

        # Crop padded regions
        reconstructed_image = reconstructed_image.crop((0, 0, original_w, original_h))

        # Save final output image
        reconstructed_image.save(output_path)

        print("\n" + "="*50)
        print("DECOMPRESSION COMPLETE")
        print("="*50)
        print(f"Reconstructed Image Saved: {output_path}")
        print("="*50)

        return True

def main():
    print("\n" + "="*60)
    print("Simple Vector Quantizer - Image Compression")
    print("="*60)
    
    while True:
        print("\nMAIN MENU")
        print("1. Compress Image")
        print("2. Decompress Image")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            print("\n--- COMPRESSION ---")
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print(" Image not found!")
                continue
            
            try:
                block_h = int(input("Enter block height: "))
                block_w = int(input("Enter block width: "))
                num_codewords = int(input("Enter number of codewords: "))
                if block_h <= 0 or block_w <= 0 or num_codewords <= 0:
                    print(" All values must be positive!")
                    continue
                
                vq = VectorQuantizer(block_h, block_w, num_codewords)
                success, ratio = vq.compress(image_path)
                
                if success:
                    print(f"\n Compression successful! Ratio: {ratio:.2f}:1")
                
            except ValueError:
                print(" Invalid input!")
        
        elif choice == '2':
            print("\n--- DECOMPRESSION ---")

            codebook_path = input("Enter codebook file path: ").strip()
            indices_path = input("Enter encoded indices file path: ").strip()
            output_image = input("Enter output image name (e.g. output.png): ").strip()

            vq = VectorQuantizer(1, 1, 1)  # dummy, values replaced by loaded metadata

            success = vq.decompress(codebook_path, indices_path, output_image)

            if success:
                print("\n Decompression done! Image saved.")

        else:
            print("Goodbye. Have a good day :)")
            return
        

if __name__ == "__main__":
    main()
