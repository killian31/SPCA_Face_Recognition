import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

"""
Sources about spca and robustness:
- https://arxiv.org/abs/2011.06585
- https://proceedings.mlr.press/v125/awasthi20a.html
- https://arxiv.org/abs/2411.05332  
"""


def load_mnist(n_samples=None):
    print("Fetching MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print("Scaling and splitting data...")
    if n_samples is None:
        n_samples = len(X)
    random_indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[random_indices]
    y = y[random_indices].astype(np.int32)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=29
    )
    return X_train, X_test, y_train, y_test, n_samples


class FixedLinear(nn.Module):
    """
    Implements a fixed linear transform using precomputed weights and bias.
    This is used to apply the PCA or SPCA transformation.
    """

    def __init__(self, weight, bias):
        super(FixedLinear, self).__init__()
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        self.linear = nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        # Freeze parameters
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x)


class FixedScaler(nn.Module):
    """
    Implements a fixed StandardScaler: (x - mean) / scale.
    """

    def __init__(self, mean, scale):
        super(FixedScaler, self).__init__()
        # Register buffers so they move with the model but are not trained.
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.scale


class ClassifierNN(nn.Module):
    """
    A simple two-layer neural network classifier.
    """

    def __init__(self, input_dim, num_classes=10):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PipelineModel(nn.Module):
    """
    This model takes raw (flattened) images, applies a fixed PCA/SPCA transformation,
    then fixed scaling, and finally a trainable neural network classifier.
    """

    def __init__(self, fixed_transform, fixed_scaler, classifier):
        super(PipelineModel, self).__init__()
        self.fixed_transform = fixed_transform  # PCA or SPCA layer
        self.fixed_scaler = fixed_scaler  # Scaler layer
        self.classifier = classifier  # Trainable NN classifier

    def forward(self, x):
        x = self.fixed_transform(x)
        x = self.fixed_scaler(x)
        x = self.classifier(x)
        return x


def train_pipeline_model(
    model, X_train, y_train, device, num_epochs=10, batch_size=128, verbose=False
):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        avg_loss = epoch_loss / len(loader.dataset)
        if verbose:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
    return model, optimizer, criterion


def setup_pipeline_classifier(
    transformer, X_train, y_train, device=torch.device("cpu"), num_epochs=20
):
    """
    Given a transformer (PCA or SparsePCA) and raw training data,
    fit the transformer and a StandardScaler on the transformed data,
    then build a composite PyTorch model that applies the fixed transformation,
    scaling, and a small NN classifier.
    The model is trained and wrapped with ART's PyTorchClassifier.
    """
    transformer.fit(X_train)
    X_train_trans = transformer.transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_trans)

    n_comp = X_train_trans.shape[1]

    # For PCA, transformer.mean_ exists; for SparsePCA, it may not.
    if hasattr(transformer, "mean_"):
        mean_val = transformer.mean_
    else:
        mean_val = np.zeros(X_train.shape[1])  # No centering for SparsePCA

    # Compute fixed transform parameters.
    # The transformation is: (x - mean) dot components_.T.
    # Thus, weight = transformer.components_ and bias = - (mean dot components_.T)
    weight = torch.tensor(transformer.components_, dtype=torch.float32)
    bias = -torch.matmul(
        torch.tensor(mean_val, dtype=torch.float32),
        torch.tensor(transformer.components_.T, dtype=torch.float32),
    )

    fixed_transform = FixedLinear(weight, bias)
    fixed_scaler = FixedScaler(scaler.mean_, scaler.scale_)

    classifier_nn = ClassifierNN(input_dim=n_comp, num_classes=10)
    model = PipelineModel(fixed_transform, fixed_scaler, classifier_nn).to(device)

    model, optimizer, criterion = train_pipeline_model(
        model, X_train, y_train, device, num_epochs=num_epochs, verbose=False
    )

    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(784,),
        nb_classes=10,
        clip_values=(0, 1),
    )
    return art_classifier


def generate_adversarial_samples(classifier, X, method="FGSM", eps=0.3):
    if method == "FGSM":
        attack = FastGradientMethod(classifier, eps=eps, batch_size=128)
    elif method == "PGD":
        attack = ProjectedGradientDescent(classifier, eps=eps, batch_size=128)
    else:
        raise ValueError(f"Invalid attack method: {method}")

    X_adv = attack.generate(x=X)
    X_adv = np.clip(X_adv, 0, 1)
    return X_adv


def evaluate_robustness(X_clean, X_adv, y, classifier):
    clean_predictions = classifier.predict(X_clean)
    adv_predictions = classifier.predict(X_adv)

    clean_predictions = np.argmax(clean_predictions, axis=1)
    adv_predictions = np.argmax(adv_predictions, axis=1)

    y = y.astype(int)
    clean_predictions = clean_predictions.astype(int)
    adv_predictions = adv_predictions.astype(int)

    clean_accuracy = np.mean(clean_predictions == y) * 100
    adv_accuracy = np.mean(adv_predictions == y) * 100

    return clean_accuracy, adv_accuracy


def benchmark_robustness(
    X_test,
    y_test,
    classifier_pca_dict,
    classifier_spca_dict,
    eps_list,
    n_samples,
    save_samples=True,
):
    pca_clean_acc_dict = {}
    spca_clean_acc_dict = {}

    for n_comp in classifier_pca_dict.keys():
        pca_clean_acc, _ = evaluate_robustness(
            X_test, X_test, y_test, classifier_pca_dict[n_comp]
        )
        spca_clean_acc, _ = evaluate_robustness(
            X_test, X_test, y_test, classifier_spca_dict[n_comp]
        )
        pca_clean_acc_dict[n_comp] = pca_clean_acc
        spca_clean_acc_dict[n_comp] = spca_clean_acc

    pca_accuracies_dict = {n_comp: [acc] for n_comp, acc in pca_clean_acc_dict.items()}
    spca_accuracies_dict = {
        n_comp: [acc] for n_comp, acc in spca_clean_acc_dict.items()
    }

    pbar = tqdm(
        total=len(eps_list) * len(classifier_pca_dict.keys()),
        desc="Testing epsilon values",
    )

    if save_samples:
        n_components_list = list(classifier_pca_dict.keys())
        X_adv_pcas = {}
        X_adv_spcas = {}

    for n_comp in classifier_pca_dict.keys():
        if save_samples:
            X_adv_pcas[n_comp] = []
            X_adv_spcas[n_comp] = []
        for eps in eps_list:
            pbar.set_description(f"Testing epsilon={eps}, n_comp={n_comp}")
            X_adv_pca = generate_adversarial_samples(
                classifier_pca_dict[n_comp], X_test, method="FGSM", eps=eps
            )
            X_adv_spca = generate_adversarial_samples(
                classifier_spca_dict[n_comp], X_test, method="FGSM", eps=eps
            )

            _, pca_adv_acc = evaluate_robustness(
                X_test, X_adv_pca, y_test, classifier_pca_dict[n_comp]
            )
            _, spca_adv_acc = evaluate_robustness(
                X_test, X_adv_spca, y_test, classifier_spca_dict[n_comp]
            )

            pca_accuracies_dict[n_comp].append(pca_adv_acc)
            spca_accuracies_dict[n_comp].append(spca_adv_acc)

            if save_samples:
                X_adv_pcas[n_comp].append(X_adv_pca[0])
                X_adv_spcas[n_comp].append(X_adv_spca[0])

            pbar.update(1)
    pbar.close()
    if save_samples:
        n_components_list = list(classifier_pca_dict.keys())
        directory = f"adv_samples_eps_{eps_list[0]}_to_{eps_list[-1]}_ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}"
        os.makedirs(directory, exist_ok=True)
        X_image = X_test[0]
        for ncp in n_components_list:
            show_adversarial_samples(
                X_image, X_adv_pcas[ncp], X_adv_spcas[ncp], eps_list, ncp, directory
            )

    return np.concatenate(([0], eps_list)), pca_accuracies_dict, spca_accuracies_dict


def plot_benchmark(
    epsilons, pca_accuracies_dict, spca_accuracies_dict, n_components_list, n_samples
):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(n_components_list), max(n_components_list))

    for n_comp in n_components_list:
        color = cmap(norm(n_comp))
        ax.plot(
            epsilons,
            pca_accuracies_dict[n_comp],
            "--o",
            color=color,
            label="PCA" if n_comp == n_components_list[0] else "_nolegend_",
        )
        ax.plot(
            epsilons,
            spca_accuracies_dict[n_comp],
            "-o",
            color=color,
            label="SPCA" if n_comp == n_components_list[0] else "_nolegend_",
        )

    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy vs. Adversarial Attack Strength (FGSM)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Number of Components")
    ax.legend()
    ax.grid(True)
    plt.savefig(
        f"fgsm_epsilons_{epsilons[1]}_to_{epsilons[-1]}_ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}.png",
        dpi=300,
    )


def show_adversarial_samples(X, X_adv_pcas, X_adv_spcas, eps_list, ncp, directory):
    """
    Show original image and its adversarial counterparts side by side for different epsilon.
    """
    n_eps = len(X_adv_pcas)
    fig, axes = plt.subplots(2, n_eps + 1, figsize=(15, 6))
    # Plot original image
    axes[0, 0].imshow(X.reshape(28, 28), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original Image")
    axes[1, 0].imshow(X.reshape(28, 28), cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original Image")
    for i, (X_adv_pca, X_adv_spca) in enumerate(zip(X_adv_pcas, X_adv_spcas)):
        axes[0, i + 1].imshow(X_adv_pca.reshape(28, 28), cmap="gray")
        axes[0, i + 1].axis("off")
        axes[0, i + 1].set_title(f"ε={eps_list[i]:.2f}")
        axes[1, i + 1].imshow(X_adv_spca.reshape(28, 28), cmap="gray")
        axes[1, i + 1].axis("off")
        axes[1, i + 1].set_title(f"ε={eps_list[i]:.2f}")
    axes[0, 0].set_ylabel("PCA")
    axes[1, 0].set_ylabel("SPCA")
    plt.tight_layout()
    plt.savefig(
        f"{directory}/adversarial_samples_{eps_list[0]:.2f}_to_{eps_list[-1]:.2f}_{ncp}.png",
        dpi=300,
    )


def main(eps_list, n_components_list, n_samples, save_samples=True):
    X_train, X_test, y_train, y_test, n_samples = load_mnist(n_samples=n_samples)

    X_train = ((X_train - X_train.min()) / (X_train.max() - X_train.min())).astype(
        np.float32
    )
    X_test = ((X_test - X_test.min()) / (X_test.max() - X_test.min())).astype(
        np.float32
    )

    # Set device for PyTorch (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing and training PCA and SPCA pipeline models...")
    classifier_pca_dict = {}
    classifier_spca_dict = {}

    for n_comp in tqdm(n_components_list):
        pca_transformer = PCA(n_components=n_comp)
        spca_transformer = SparsePCA(
            n_components=n_comp, random_state=29, max_iter=100, alpha=1
        )

        classifier_pca_dict[n_comp] = setup_pipeline_classifier(
            pca_transformer, X_train, y_train, device=device, num_epochs=20
        )
        classifier_spca_dict[n_comp] = setup_pipeline_classifier(
            spca_transformer, X_train, y_train, device=device, num_epochs=20
        )

    print("Running benchmark tests...")
    epsilons, pca_accuracies_dict, spca_accuracies_dict = benchmark_robustness(
        X_test,
        y_test,
        classifier_pca_dict,
        classifier_spca_dict,
        eps_list,
        n_samples,
        save_samples=save_samples,
    )

    print("\nBenchmark Results:")
    print("Epsilon values:", [round(i, 2) for i in epsilons])
    for n_comp in n_components_list:
        print(f"\nResults for {n_comp} components:")
        print(f"PCA accuracies: {[round(i, 2) for i in pca_accuracies_dict[n_comp]]}")
        print(f"SPCA accuracies: {[round(i, 2) for i in spca_accuracies_dict[n_comp]]}")

    plot_benchmark(
        epsilons,
        pca_accuracies_dict,
        spca_accuracies_dict,
        n_components_list,
        n_samples,
    )


def unit_test_benchmark():
    eps_list = np.arange(0.1, 1.0, 0.1)
    n_components_list = [150, 191]
    main(eps_list, n_components_list, 1000, save_samples=True)


if __name__ == "__main__":
    test_mode = False
    if test_mode:
        unit_test_benchmark()
        exit()
    n_components_list = [
        20,
        50,
        100,
        150,
        191,
    ]  # 191 approximates 95% cumulative explained variance
    eps_list = np.arange(0.05, 0.55, 0.05)
    n_samples = None
    main(eps_list, n_components_list, n_samples, save_samples=True)
