import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import pairwise_distances

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------
print("Loading the Olivetti Faces dataset...")

faces = fetch_olivetti_faces(shuffle=True, random_state=42)

X = np.asarray(faces.data, dtype=np.float64)
y = faces.target
images = faces.images

print("Dataset loaded successfully!")
print("Total images:", X.shape[0])
print("Features per image:", X.shape[1])
print("Number of people:", len(np.unique(y)))

# --------------------------------------------------
# 2. SPLIT INTO TRAINING AND TESTING DATA
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# --------------------------------------------------
# 3. APPLY PCA
# --------------------------------------------------
n_components = 100

pca = PCA(
    n_components=n_components,
    svd_solver="full",
    whiten=False
)

X_train_pca = pca.fit_transform(X_train)

# Project test faces onto the principal components
X_test_centered = X_test - pca.mean_

X_test_pca = np.array([
    np.sum(row * pca.components_, axis=1)
    for row in X_test_centered
])


print("\nPCA completed!")
print("Original features:", X_train.shape[1])
print("Reduced features:", X_train_pca.shape[1])

variance = np.sum(pca.explained_variance_ratio_) * 100
print(
    f"Variance retained by {n_components} components: "
    f"{variance:.2f}%"
)


# --------------------------------------------------
# 4. FACE MATCHING USING EUCLIDEAN DISTANCE
# --------------------------------------------------
# Calculate Euclidean distances without matrix multiplication
distances = np.empty((len(X_test_pca), len(X_train_pca)))

for i, test_face in enumerate(X_test_pca):
    difference = X_train_pca - test_face
    distances[i] = np.sqrt(
        np.sum(difference * difference, axis=1)
    )

nearest_indices = np.argmin(distances, axis=1)

y_pred = y_train[nearest_indices]

accuracy = accuracy_score(y_test, y_pred)

print("\nFACE MATCHING RESULTS")
print("---------------------")
print(f"Recognition accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 5. CREATE OUTPUT FOLDER
# --------------------------------------------------
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 6. VISUALIZE THE MEAN FACE AND EIGENFACES
# --------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

axes[0, 0].imshow(
    pca.mean_.reshape(64, 64), cmap="gray"
)
axes[0, 0].set_title("Mean Face")
axes[0, 0].axis("off")

for i in range(9):
    eigenface = pca.components_[i].reshape(64, 64)
    ax = axes.flat[i + 1]
    ax.imshow(eigenface, cmap="gray")
    ax.set_title(f"Eigenface {i + 1}")
    ax.axis("off")

plt.suptitle("Mean Face and Principal Components")
plt.tight_layout()
plt.savefig("outputs/eigenfaces.png", dpi=150)
plt.show()

# --------------------------------------------------
# 7. PLOT EXPLAINED VARIANCE
# --------------------------------------------------
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(9, 5))
plt.plot(
    range(1, n_components + 1),
    cumulative_variance * 100
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Retained (%)")
plt.title("PCA: Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/explained_variance.png", dpi=150)
plt.show()

# --------------------------------------------------
# 8. SHOW FACE MATCHING EXAMPLES
# --------------------------------------------------

# --------------------------------------------------
# 8. SHOW FACE MATCHING EXAMPLES
# --------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i, ax in enumerate(axes.flat):
    j = i // 2

    if i % 2 == 0:
        # Display the test image
        ax.imshow(X_test[j].reshape(64, 64), cmap="gray")
        ax.set_title(f"Test: Person {y_test[j]}")
    else:
        # Display the closest training image
        ax.imshow(
            X_train[nearest_indices[j]].reshape(64, 64),
            cmap="gray"
        )
        ax.set_title(f"Match: Person {y_pred[j]}")

    ax.axis("off")

plt.suptitle("Test Faces and Closest Training Matches")
plt.tight_layout()
plt.savefig("outputs/face_matches.png", dpi=150)
plt.show()


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------
with open("outputs/results.txt", "w") as file:
    file.write("PCA AND EIGENFACES PROJECT RESULTS\n")
    file.write(f"Total images: {len(X)}\n")
    file.write(f"Training images: {len(X_train)}\n")
    file.write(f"Testing images: {len(X_test)}\n")
    file.write(f"Principal components: {n_components}\n")
    file.write(f"Variance retained: {variance:.2f}%\n")
    file.write(f"Recognition accuracy: {accuracy * 100:.2f}%\n")

print("\nProject completed!")
print("Check the outputs folder for saved graphs and results.")
