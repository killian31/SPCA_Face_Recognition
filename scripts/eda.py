import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import os 

figures_dir = "../figures"
os.makedirs(figures_dir, exist_ok=True)

# Step 1: Load the dataset
print("Loading LFW dataset...")
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
images = lfw_dataset.images
X = lfw_dataset.data
n_samples, h, w = images.shape
target_names = lfw_dataset.target_names
n_classes = len(target_names)

print(f"Dataset loaded with {n_samples} samples.")
print(f"Image dimensions: {h}x{w}")
print(f"Number of classes: {n_classes}")
print("Classes:", target_names)

# Step 2: Visualize a few sample images
def plot_sample_images(images, target, target_names, h, w, n_row=3, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(target_names[target[i]], size=12)
        plt.xticks(())
        plt.yticks(())
    save_path = os.path.join(figures_dir, "sample_images.png")
    plt.savefig(save_path)
    plt.show()

print("Displaying sample images...")
plot_sample_images(images, lfw_dataset.target, target_names, h, w)

# Step 3: Analyze pixel intensity distribution
def plot_pixel_distribution(images):
    pixel_values = images.flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=50, color='blue', alpha=0.7)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    save_path = os.path.join(figures_dir, "pixel_intensity.png")
    plt.savefig(save_path)
    plt.show()

print("Analyzing pixel intensity distribution...")
plot_pixel_distribution(images)

# Step 4: Class distribution
def plot_class_distribution(target, target_names):
    class_counts = np.bincount(target)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(target_names)), class_counts, color='blue', alpha=0.8)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(save_path)
    plt.show()

print("Analyzing class distribution...")
plot_class_distribution(lfw_dataset.target, target_names)

# Step 5: Summary statistics for images
def compute_image_statistics(images):
    mean_image = np.mean(images, axis=0)
    std_image = np.std(images, axis=0)
    return mean_image, std_image

mean_image, std_image = compute_image_statistics(X)

# Visualize mean and standard deviation images
def plot_image_statistics(mean_image, std_image, h, w):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mean_image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title("Mean Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(std_image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title("Standard Deviation Image")
    plt.axis("off")
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "mean_sd.png")
    plt.savefig(save_path)
    plt.show()

print("Visualizing mean and standard deviation of images...")
plot_image_statistics(mean_image, std_image, h, w)