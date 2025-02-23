import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA
from sklearn.svm import SVC
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Sources about spca and robustness:
- https://arxiv.org/abs/2011.06585
- https://proceedings.mlr.press/v125/awasthi20a.html
- https://arxiv.org/abs/2411.05332  
"""

def setup_classifier(X_train, y_train):
    # Initialize and train an SVM classifier with adjusted parameters
    svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    
    # Wrap the sklearn classifier for ART with proper scaling
    classifier = SklearnClassifier(
        model=svm,
        clip_values=(0, 1),
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1)
    )
    return classifier

def load_mnist(n_samples=None):
    print("Fetching MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    print("Scaling and splitting data...")
    # Take a smaller subset for testing
    if n_samples is None:
        n_samples = len(X)
    random_indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[random_indices]
    y = y[random_indices].astype(np.int32)  # Convert labels to int32
    
    # Scale data to [0,1] range
    X = X / 255.0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)
    return X_train, X_test, y_train, y_test

def generate_adversarial_samples(classifier, X, method='FGSM', eps=0.3):
    if method == 'FGSM':
        attack = FastGradientMethod(classifier, eps=eps, batch_size=128)
    elif method == 'PGD':
        attack = ProjectedGradientDescent(classifier, eps=eps, batch_size=128)
    
    # Generate adversarial samples in batches to avoid memory issues
    X_adv = attack.generate(x=X)
    
    # Ensure the adversarial samples are within valid range
    X_adv = np.clip(X_adv, 0, 1)
    return X_adv

def evaluate_robustness(X_clean, X_adv, y, classifier):
    # Classification accuracy
    clean_predictions = classifier.predict(X_clean)
    adv_predictions = classifier.predict(X_adv)
    
    # Convert predictions to class labels by taking argmax
    clean_predictions = np.argmax(clean_predictions, axis=1)
    adv_predictions = np.argmax(adv_predictions, axis=1)
    
    # Convert string labels to integers if needed
    y = y.astype(int)
    clean_predictions = clean_predictions.astype(int)
    adv_predictions = adv_predictions.astype(int)
    
    clean_accuracy = np.mean(clean_predictions == y) * 100
    adv_accuracy = np.mean(adv_predictions == y) * 100
    
    return clean_accuracy, adv_accuracy

def benchmark_robustness(X_test_pca_dict, X_test_spca_dict, y_test, classifier_pca_dict, classifier_spca_dict, eps_list):
    # Get clean accuracies first (epsilon = 0)
    pca_clean_acc_dict = {}
    spca_clean_acc_dict = {}
    
    for n_comp in X_test_pca_dict.keys():
        pca_clean_acc, _ = evaluate_robustness(X_test_pca_dict[n_comp], X_test_pca_dict[n_comp], y_test, classifier_pca_dict[n_comp])
        spca_clean_acc, _ = evaluate_robustness(X_test_spca_dict[n_comp], X_test_spca_dict[n_comp], y_test, classifier_spca_dict[n_comp])
        pca_clean_acc_dict[n_comp] = pca_clean_acc
        spca_clean_acc_dict[n_comp] = spca_clean_acc
    
    # Initialize lists to store accuracies
    pca_accuracies_dict = {n_comp: [acc] for n_comp, acc in pca_clean_acc_dict.items()}
    spca_accuracies_dict = {n_comp: [acc] for n_comp, acc in spca_clean_acc_dict.items()}
    
    # Test different epsilon values
    pbar = tqdm(total=len(eps_list) * len(X_test_pca_dict.keys()), desc='Testing epsilon values')
    for eps in eps_list:
        for n_comp in X_test_pca_dict.keys():
            pbar.set_description(f'Testing epsilon={eps}, n_comp={n_comp}')
            # Generate adversarial samples for both methods
            X_test_pca_adv = generate_adversarial_samples(classifier_pca_dict[n_comp], X_test_pca_dict[n_comp], method='FGSM', eps=eps)
            X_test_spca_adv = generate_adversarial_samples(classifier_spca_dict[n_comp], X_test_spca_dict[n_comp], method='FGSM', eps=eps)
            
            # Evaluate robustness
            _, pca_adv_acc = evaluate_robustness(X_test_pca_dict[n_comp], X_test_pca_adv, y_test, classifier_pca_dict[n_comp])
            _, spca_adv_acc = evaluate_robustness(X_test_spca_dict[n_comp], X_test_spca_adv, y_test, classifier_spca_dict[n_comp])
            
            pca_accuracies_dict[n_comp].append(pca_adv_acc)
            spca_accuracies_dict[n_comp].append(spca_adv_acc)
            pbar.update(1)
    pbar.close()
    
    return np.concatenate(([0], eps_list)), pca_accuracies_dict, spca_accuracies_dict

def main(eps_list, n_components_list, n_samples):
    # Load and preprocess data
    print("Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist(n_samples=n_samples)
    
    # Scale data to [0,1] range for adversarial attacks
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    
    print("Initializing and fitting dimensionality reduction methods...")
    
    # Dictionaries to store transformed data and classifiers
    X_train_pca_dict = {}
    X_test_pca_dict = {}
    X_train_spca_dict = {}
    X_test_spca_dict = {}
    classifier_pca_dict = {}
    classifier_spca_dict = {}
    
    for n_comp in tqdm(n_components_list):
        # Initialize and fit PCA and SPCA
        pca = PCA(n_components=n_comp)
        sparse_pca = SparsePCA(n_components=n_comp, random_state=29, max_iter=100, alpha=1.0)
        
        # Fit transformations
        pca.fit(X_train)
        sparse_pca.fit(X_train)
        
        # Transform data
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_train_spca = sparse_pca.transform(X_train)
        X_test_spca = sparse_pca.transform(X_test)
        
        # Normalize transformed data
        scaler = StandardScaler()
        X_train_pca = scaler.fit_transform(X_train_pca)
        X_test_pca = scaler.transform(X_test_pca)
        X_train_spca = scaler.fit_transform(X_train_spca)
        X_test_spca = scaler.transform(X_test_spca)
        
        # Store transformed data
        X_train_pca_dict[n_comp] = X_train_pca
        X_test_pca_dict[n_comp] = X_test_pca
        X_train_spca_dict[n_comp] = X_train_spca
        X_test_spca_dict[n_comp] = X_test_spca
        
        # Setup classifiers
        classifier_pca_dict[n_comp] = setup_classifier(X_train_pca, y_train)
        classifier_spca_dict[n_comp] = setup_classifier(X_train_spca, y_train)
    
    # Run benchmark tests
    print("Running benchmark tests...")
    epsilons, pca_accuracies_dict, spca_accuracies_dict = benchmark_robustness(
        X_test_pca_dict, X_test_spca_dict, y_test, classifier_pca_dict, classifier_spca_dict, eps_list
    )
    # Print final results
    print("\nBenchmark Results:")
    print("Epsilon values:", epsilons)
    for n_comp in n_components_list:
        print(f"\nResults for {n_comp} components:")
        print(f"PCA accuracies: {pca_accuracies_dict[n_comp]}")
        print(f"SPCA accuracies: {spca_accuracies_dict[n_comp]}")
    
    # Create and save the plot with enhanced visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(n_components_list), max(n_components_list))
    
    # Plot lines for each number of components
    for n_comp in n_components_list:
        color = cmap(norm(n_comp))
        # Plot PCA (dashed lines)
        ax.plot(epsilons, pca_accuracies_dict[n_comp], '--o', color=color, label=f'PCA ({n_comp} components)')
        # Plot SPCA (solid lines)
        ax.plot(epsilons, spca_accuracies_dict[n_comp], '-o', color=color, label=f'SPCA ({n_comp} components)')
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy vs. Adversarial Attack Strength (FGSM)')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Number of Components')
    
    # Add legend
    ax.legend()
    ax.grid(True)
    plt.savefig('robustness_comparison.png')
    plt.show()

if __name__ == "__main__":
    n_components_list = [20, 50, 100, 150, 191] # 191 = 95% of cumulative explained variance
    eps_list = np.arange(0.1, 1.0, 0.1)
    n_samples = 5000
    main(eps_list, n_components_list, n_samples)