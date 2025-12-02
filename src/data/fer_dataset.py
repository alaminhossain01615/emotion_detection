import numpy as np
import os
import matplotlib.pyplot as plt

class ImageDatasetExplorer:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_counts = {}
        self.class_names = []
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    def explore_data(self):
        # Get all subdirectories (which we assume are the classes)
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
        print("Data exploration complete.")
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
        

        # Add counts on top of the bars for clarity
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), 
                     ha='center', va='bottom', fontsize=10)

        plt.title('Image Count Distribution by Class', fontsize=16)
        plt.xlabel('Class Name', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"\nPlot saved to {save_path}")
        else:
            plt.show()
            print("\nPlot displayed.")

    def show_some_random_data_samples(self):

        # Implement later on
