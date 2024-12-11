import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA, MiniBatchSparsePCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the LFW dataset
print("Loading LFW dataset...")
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_dataset.data
n_samples, h, w = lfw_dataset.images.shape

print(f"Dataset loaded with {n_samples} images of size {h}x{w}.")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: PCA Implementation
print("Performing PCA...")
n_components = 15
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
pca_components = pca.components_

# Step 3: Custom SSPCA Implementation
class SSPCA:
    def __init__(self, n_components, alpha=0.1, max_iter=50, group_size=5, tol=1e-6, verbose=False):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.group_size = group_size
        self.tol = tol
        self.verbose = verbose

    def fit_transform(self, X):
        pca = PCA(n_components=self.n_components)
        Z = pca.fit_transform(X)  # PCA initialization
        components = pca.components_
        for iteration in range(self.max_iter):
            old_components = components.copy()
            components = self._apply_sparsity(components)
            Z = X @ components.T  # Update latent representation
            components = (Z.T @ X) / (np.linalg.norm(Z.T @ X, axis=1, keepdims=True) + 1e-8)

            # Check for convergence
            diff = np.linalg.norm(components - old_components)
            if self.verbose:
                print(f"Iteration {iteration + 1}, component diff = {diff}")
            if diff < self.tol:
                if self.verbose:
                    print("Converged early.")
                break
        self.components_ = components
        return Z

    def _apply_sparsity(self, components):
        for i in range(components.shape[0]):  # For each principal component
            for start in range(0, components.shape[1], self.group_size):
                end = min(start + self.group_size, components.shape[1])
                group = components[i, start:end]
                group_norm = np.linalg.norm(group)
                if group_norm <= self.alpha:
                    # Zero out the entire group
                    components[i, start:end] = 0.0
                else:
                    # Shrink the group
                    shrink_factor = 1 - self.alpha / group_norm
                    components[i, start:end] = group * shrink_factor
        return components

print("Performing personalized SSPCA...")
sspca = SSPCA(n_components=n_components, alpha=0.1)
X_sspca = sspca.fit_transform(X_scaled)
sspca_components = sspca.components_

print("Performing Mini Batch SSPCA...")
mb_sspca = MiniBatchSparsePCA(n_components=n_components, alpha=0.1, batch_size=100, max_iter=50)
X_mb_sspca = mb_sspca.fit_transform(X_scaled)
mb_sspca_components = mb_sspca.components_

# Step 4: Visualization of Components
def plot_gallery(title, images, n_col=5, n_row=3, image_shape=(h, w)):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:n_col * n_row]):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()

print("Visualizing Original Images...")
plot_gallery("Original Images", X)

print("Visualizing PCA components...")
plot_gallery("PCA Components", pca_components, image_shape=(h, w))

print("Visualizing personalized SSPCA components...")
plot_gallery("SSPCA Components", sspca_components, image_shape=(h, w))

print("Visualizing Mini Batch SSPCA components...")
plot_gallery("Mini Batch SSPCA Components", mb_sspca_components, image_shape=(h, w))

# Step 5: Reconstruction Error Comparison
X_pca_reconstructed = pca.inverse_transform(X_pca)
X_sspca_reconstructed = X_scaled @ sspca_components.T @ sspca_components
X_mb_sspca_reconstructed = X_scaled @ mb_sspca_components.T @ mb_sspca_components

pca_reconstruction_error = np.mean((X_scaled - X_pca_reconstructed) ** 2)
sspca_reconstruction_error = np.mean((X_scaled - X_sspca_reconstructed) ** 2)
mb_sspca_reconstruction_error = np.mean((X_scaled - X_mb_sspca_reconstructed) ** 2)

print(f"PCA Reconstruction Error: {pca_reconstruction_error:.4f}")
print(f"Personalized SSPCA Reconstruction Error: {sspca_reconstruction_error:.4f}")
print(f"Mini Batch SSPCA Reconstruction Error: {mb_sspca_reconstruction_error:.4f}")