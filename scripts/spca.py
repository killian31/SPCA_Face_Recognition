# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA
from sklearn.preprocessing import StandardScaler
from eda import X

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Performing PCA...")
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
pca_components = pca.components_

print(f"Shape of X_scaled: {X_scaled.shape}")  # Standardized features
print(f"Shape of X_pca: {X_pca.shape}")  # Reduced features

# Define Sparse PCA with desired number of components
n_components = 171 
sparse_pca = SparsePCA(n_components=n_components, random_state=42, alpha=1)

# Fit and transform the data
X_sparse_pca = sparse_pca.fit_transform(X_scaled)
sparse_pca_components = sparse_pca.components_

# Print shapes
print(f"Shape of X_scaled: {X_scaled.shape}")  # Standardized features
print(f"Shape of X_sparse_pca: {X_sparse_pca.shape}")  # Reduced features

# Define Mini Batch Sparse PCA with desired number of components

mbsparse_pca = MiniBatchSparsePCA(n_components=n_components, alpha=0.1, batch_size=100, max_iter=50, random_state=42)

# Fit and transform the data
X_mbsparse_pca = mbsparse_pca.fit_transform(X_scaled)
mbsparse_pca_components = mbsparse_pca.components_

# Print shapes
print(f"Shape of X_scaled: {X_scaled.shape}")  # Standardized features
print(f"Shape of X_sparse_pca: {X_mbsparse_pca.shape}")  # Reduced features

# Visualize the first few PCA components as images
n_visualize = 5  # Number of components to visualize
plt.figure(figsize=(15, 5))

for i in range(n_visualize):
    plt.subplot(1, n_visualize, i + 1)
    plt.imshow(pca_components[i].reshape(h, w), cmap='gray')
    plt.title(f"PCA Component {i+1}")
    plt.axis('off')

plt.show()

n_visualize = 5  # Number of components to visualize
plt.figure(figsize=(15, 5))

for i in range(n_visualize):
    plt.subplot(1, n_visualize, i + 1)
    plt.imshow(sparse_pca_components[i].reshape(h, w), cmap='gray')
    plt.title(f"Sparse PCA Component {i+1}")
    plt.axis('off')

plt.show()

n_visualize = 5  # Number of components to visualize
plt.figure(figsize=(15, 5))

for i in range(n_visualize):
    plt.subplot(1, n_visualize, i + 1)
    plt.imshow(mbsparse_pca_components[i].reshape(h, w), cmap='gray')
    plt.title(f"Sparse PCA Component {i+1}")
    plt.axis('off')

plt.show()

# 2D PCA visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=lfw_dataset.target, cmap='rainbow', alpha=0.7, edgecolor='k')
plt.colorbar(ticks=range(len(target_names)), label="Class")
plt.title("PCA Projection (2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()

# 2D Sparse PCA visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_sparse_pca[:, 0], X_sparse_pca[:, 1], c=lfw_dataset.target, cmap='rainbow', alpha=0.7, edgecolor='k')
plt.colorbar(ticks=range(len(target_names)), label="Class")
plt.title("Sparse PCA Projection (2D)")
plt.xlabel("Sparse PCA Component 1")
plt.ylabel("Sparse PCA Component 2")
plt.grid()
plt.show()

# 2D Mini-batch Sparse PCA visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_mbsparse_pca[:, 0], X_mbsparse_pca[:, 1], c=lfw_dataset.target, cmap='rainbow', alpha=0.7, edgecolor='k')
plt.colorbar(ticks=range(len(target_names)), label="Class")
plt.title("Mini-batch Sparse PCA Projection (2D)")
plt.xlabel("Sparse PCA Component 1")
plt.ylabel("Sparse PCA Component 2")
plt.grid()
plt.show()

# 3D PCA visualization (if n_components >= 3)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=lfw_dataset.target, cmap='rainbow', alpha=0.7)
plt.colorbar(scatter, label="Class")
ax.set_title("PCA Projection (3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.show()

# 3D Sparse PCA visualization (if n_components >= 3)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_sparse_pca[:, 0], X_pca[:, 1], X_sparse_pca[:, 2], c=lfw_dataset.target, cmap='rainbow', alpha=0.7)
plt.colorbar(scatter, label="Class")
ax.set_title("SPCA Projection (3D)")
ax.set_xlabel("SPCA Component 1")
ax.set_ylabel("SPCA Component 2")
ax.set_zlabel("SPCA Component 3")
plt.show()

# 3D MB Sparse PCA visualization (if n_components >= 3)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_mbsparse_pca[:, 0], X_pca[:, 1], X_mbsparse_pca[:, 2], c=lfw_dataset.target, cmap='rainbow', alpha=0.7)
plt.colorbar(scatter, label="Class")
ax.set_title("Mini-batch SPCA Projection (3D)")
ax.set_xlabel("Mini-batch SPCA Component 1")
ax.set_ylabel("Mini-batch SPCA Component 2")
ax.set_zlabel("Mini-batch SPCA Component 3")
plt.show()