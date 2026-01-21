import numpy as np
import os
import matplotlib.pyplot as plt
import random
from PIL import Image

class ImageDatasetExplorer:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_counts = {}
        self.class_names = []
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    def explore_data(self):
        all_entries = os.listdir(self.root_dir)
        self.class_names = sorted([
            name for name in all_entries 
            if os.path.isdir(os.path.join(self.root_dir, name))
        ])

        if not self.class_names:
            print("Error: No class subfolders found in the root directory.")
            return

        # Count images in each class folder
        for class_name in self.class_names:
            class_path = os.path.join(self.root_dir, class_name)
            count = 0
            # Iterate through all files in the class folder
            for filename in os.listdir(class_path):
                if filename.lower().endswith(self.image_extensions):
                    count += 1
            self.class_counts[class_name] = count
        self.print_summary()

    def print_summary(self):
        print("\n--- Dataset Summary ---")
        total_images = sum(self.class_counts.values())
        print(f"Total Classes Found: {len(self.class_names)}")
        print(f"Total Images Found: {total_images}")
        
        print("\nClass-wise Image Counts:")
        for class_name, count in self.class_counts.items():
            print(f"- **{class_name}**: {count} images")
        
        # Calculate and print data imbalance metric
        if self.class_counts:
            counts = list(self.class_counts.values())
            min_count = min(counts)
            max_count = max(counts)
            print(f"\nData Imbalance Ratio (Max/Min): {max_count/min_count:.2f}")

    def plot_class_distribution(self, save_path=None):

        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, counts, color='skyblue') 
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), 
                     ha='center', va='bottom', fontsize=14)

        plt.title('Image Count Distribution by Class', fontsize=16)
        plt.xlabel('Class Name', fontsize=16)
        plt.ylabel('Number of Images', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"\nPlot saved to {save_path}")
        else:
            plt.show()
            print("\nPlot displayed.")

    def display_some_random_samples(self):
        
        sample_images=[]

        for class_name in self.class_names:
            class_path = os.path.join(self.root_dir,class_name)
            image_files=[file for file in os.listdir(class_path) if file.lower().endswith(self.image_extensions)]

            if image_files:
                rndm_img_file=random.choice(image_files)
                rndm_img_path=os.path.join(class_path,rndm_img_file)
                sample_images.append((class_name,rndm_img_path))
            else:
                print(f"No image in the class folder named {class_name}")

        if not sample_images:
            print(f"No images found")

        grid_cols=min(len(sample_images),4)
        grid_rows=int(np.ceil(len(sample_images)/grid_cols))

        plt.figure(figsize=(3*grid_cols, 3*grid_rows))

        for i,(class_name,image_path) in enumerate(sample_images):
            img=Image.open(image_path).convert("RGB")
            sp = plt.subplot(grid_rows,grid_cols,i+1)
            sp.imshow(img)
            sp.set_title(class_name, fontsize=16)
            sp.axis("off")

        plt.show()