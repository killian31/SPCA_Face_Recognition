
# Sparse Principal Component Analysis (Sparse PCA)

Sparse Principal Component Analysis (Sparse PCA) is an advanced machine learning technique that combines the power of dimensionality reduction with the interpretability of sparsity. Unlike traditional PCA, Sparse PCA generates components that are sparse, meaning they depend only on a subset of the original features. This makes Sparse PCA particularly useful for feature selection and data interpretation.

---

## 🌟 Key Features

- **Sparsity**: Identifies only the most relevant features in the data.
- **Interpretability**: Enhances the clarity of results by focusing on fewer variables.
- **Flexibility**: Allows customization of sparsity levels via regularization parameters.
- **Robust Dimensionality Reduction**: Reduces data complexity while retaining essential information.

---

## 🛠️ How It Works

Sparse PCA minimizes the reconstruction error with sparsity constraints on the principal components. The optimization problem can be formulated as:

$$
\min_{W, H} \left\Vert X - W H^T \right\Vert_F^2 + \alpha \left\Vert H \right\Vert_1 + \beta \left\Vert H\right\Vert_2^2
$$

Where:
- $X$: Input data matrix.
- $W$: Principal components matrix.
- $H$: Sparse loadings matrix.
- $\alpha$: Controls sparsity (L1 regularization).
- $\beta$: Controls smoothness (L2 regularization).

The algorithm alternates between:
1. Fixing $H$ to optimize $W$.
2. Fixing $W$ to optimize $H$ using Elastic Net regularization.

---

## 🚀 Quickstart Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/theodruilhe/Sparse_PCA.git
cd Sparse_PCA
```

### 2️⃣ Install Dependencies

Ensure you have Python installed along with the required libraries:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Sparse PCA

Here's an example of how to use Sparse PCA on your dataset:

```python
import numpy as np
from sparse_pca import SparsePCA

# Load your dataset
X = np.loadtxt('your_data_file.txt')

# Initialize the Sparse PCA model
model = SparsePCA(n_components=5, alpha=0.1, beta=0.1)

# Fit the model
model.fit(X)

# Transform your data
X_transformed = model.transform(X)

# Get the components
components = model.components_

print("Sparse Components:", components)
```

---

## 📚 Documentation

### Parameters
- **`n_components`**: Number of principal components to compute.
- **`alpha`**: Sparsity regularization parameter (L1 norm).
- **`beta`**: Ridge regularization parameter (L2 norm).

### Outputs
- **`components_`**: Sparse principal components.
- **`explained_variance_`**: Variance explained by each component.

---

## 📊 Example Use Case

Suppose you have a high-dimensional dataset with 1,000 features, and you suspect that only a handful of these features are relevant. Using Sparse PCA, you can:
1. Reduce the dataset's dimensionality.
2. Identify the most critical features for your problem.

---

## 📝 References

- Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse Principal Component Analysis. *Journal of Computational and Graphical Statistics*, 15(2), 265-286.
- Jenatton, R., Obozinski, G., & Bach, F. (2009). Structured Sparse Principal Component Analysis. [arXiv preprint](https://arxiv.org/abs/0909.1440).

---

## 💻 Contributions

We welcome contributions to this project! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

Special thanks to the creators of Sparse PCA algorithms and the open-source community for their invaluable contributions to data science.
