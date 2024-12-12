from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from eda import target_names, n_classes, lfw_dataset
from spca import X_scaled, pca, sparse_pca, mbsparse_pca



# Create a train set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, lfw_dataset.target, test_size=0.2, random_state=42, shuffle=True, stratify=None
)

# Transform the data using PCA
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Transform the data using Sparse PCA
X_train_sparse_pca = sparse_pca.fit_transform(X_train)
X_test_sparse_pca = sparse_pca.transform(X_test)

# Transform the data using MiniBatchSparsePCA
X_train_mbsparse_pca = mbsparse_pca.fit_transform(X_train)
X_test_mbsparse_pca = mbsparse_pca.transform(X_test)

# Define the parameter grid
param_grid = {
    'C': [1, 5, 10, 20],  # Test various regularization strengths
    'gamma': ['scale', 0.1, 0.01, 0.001]  # Test different gamma values
}

# Instantiate the SVC model
svm = SVC(kernel='rbf', random_state=42)

# Setup the GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Results for raw data:")

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.4f}')

# Use the best parameters to make predictions on the test set
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data with tuned parameters: {acc:.4f}')

# print classifiction results 
print(classification_report(y_test, y_pred, target_names = target_names)) 
# print confusion matrix 
print("Confusion Matrix is:") 
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))


grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train_pca, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Results for PCA:")

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.4f}')

# Use the best parameters to make predictions on the test set
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data with tuned parameters: {acc:.4f}')

# print classifiction results 
print(classification_report(y_test, y_pred, target_names = target_names)) 
# print confusion matrix 
print("Confusion Matrix is:") 
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train_sparse_pca, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Results for SPCA:")

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.4f}')

# Use the best parameters to make predictions on the test set
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_sparse_pca)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data with tuned parameters: {acc:.4f}')

# print classifiction results 
print(classification_report(y_test, y_pred, target_names = target_names)) 
# print confusion matrix 
print("Confusion Matrix is:") 
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train_mbsparse_pca, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Results for Mini Batch SPCA:")

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.4f}')

# Use the best parameters to make predictions on the test set
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_mbsparse_pca)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data with tuned parameters: {acc:.4f}')

# print classifiction results 
print(classification_report(y_test, y_pred, target_names = target_names)) 
# print confusion matrix 
print("Confusion Matrix is:") 
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))