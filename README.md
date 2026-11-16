# Face Recognition using Eigenvalues and Principal Component Analysis (PCA)

An academic and interactive machine learning laboratory project implementing **Principal Component Analysis (PCA)** and the classical **Eigenfaces** algorithm for facial recognition on the AT&T Cambridge Olivetti Faces dataset.

---

## 📌 Project Introduction

Face recognition is one of the most prominent applications of linear algebra and machine learning in computer vision. High-dimensional face images contain substantial redundancy due to regular facial geometries (eyes, nose, mouth positions) and illumination similarities. 

This project demonstrates how **Principal Component Analysis (PCA)** can drastically reduce this high dimensionality by projecting face images into an optimal low-dimensional orthogonal subspace termed **"Face Space"**. The basis vectors that span this subspace are eigenvectors of the image covariance matrix, known conceptually as **Eigenfaces**. Unknown probe faces are projected onto these basis vectors and classified against training samples using Euclidean distance ($L_2$ norm) minimization.

---

## 🎯 Project Objectives

1. **Dimensionality Reduction**: Compress high-dimensional image vectors ($64 \times 64 = 4,096$ pixels) down to a compact 100-dimensional subspace while retaining over **94%** of image variance.
2. **Covariance Matrix Decomposition**: Compute eigenvalues ($\lambda$) and eigenvectors ($\mathbf{v}$) representing primary axes of facial variance.
3. **Facial Feature Extraction**: Generate and visualize the global **Mean Face ($\Psi$)** and principal **Eigenfaces ($\mathbf{u}_k$)**.
4. **Automated Recognition**: Classify unknown probe faces using $1$-Nearest Neighbor Euclidean distance matching in the reduced feature space.
5. **Interactive Visualization**: Provide an interactive laboratory application (`index.html`) allowing real-time parameter exploration, distance calculations, and step-by-step vector projections.

---

## 🧠 Eigenfaces and PCA: Working Methodology

The mathematical methodology follows the foundational framework formulated by Sirovich & Kirby (1987) and Turk & Pentland (1991):

### 1. Vector Representation
Each $M \times N$ face image is flattened into a 1D column vector $\Gamma_i \in \mathbb{R}^{D}$, where $D = M \times N = 64 \times 64 = 4,096$.

### 2. Mean Face Computation
The global average face vector $\Psi$ across $M$ training samples is:
$$\Psi = \frac{1}{M} \sum_{i=1}^{M} \Gamma_i$$

### 3. Mean-Centering
Each training image is centered by subtracting the mean face:
$$\Phi_i = \Gamma_i - \Psi$$
All centered vectors form the data matrix $A = [\Phi_1, \Phi_2, \dots, \Phi_M] \in \mathbb{R}^{D \times M}$.

### 4. Covariance & Eigen-Decomposition
The sample covariance matrix is:
$$C = \frac{1}{M} A A^T \in \mathbb{R}^{D \times D}$$
Since $D = 4,096 \gg M = 300$, scikit-learn computes the eigenvectors via Singular Value Decomposition (SVD) of $A$, yielding the top $K=100$ principal components $\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_K$ corresponding to the largest eigenvalues.

### 5. Subspace Projection
Each mean-centered face $\Phi$ is projected into the $K$-dimensional subspace:
$$\omega_k = \mathbf{u}_k^T \Phi = \mathbf{u}_k^T (\Gamma - \Psi)$$
$$\mathbf{\Omega} = [\omega_1, \omega_2, \dots, \omega_K]^T \in \mathbb{R}^{K}$$

### 6. Classification via Euclidean Distance
For a test face vector $\mathbf{\Omega}_{test}$, the Euclidean distance $\epsilon_k$ to every training face weight vector $\mathbf{\Omega}_k$ is computed:
$$\epsilon_k = \|\mathbf{\Omega}_{test} - \mathbf{\Omega}_k\|_2 = \sqrt{\sum_{j=1}^{K} (\omega_{test, j} - \omega_{k, j})^2}$$
The identified identity corresponds to:
$$k^* = \arg\min_k \epsilon_k$$

---

## ✨ Key Features

- **Modular & Clean Architecture**: Cleanly separated HTML (`index.html`), CSS (`styles.css`), and JavaScript (`script.js`), ensuring all source code is lightweight and viewable directly on GitHub without truncation.
- **Standardized PCA Pipeline**: Pure scikit-learn and NumPy implementation for dataset loading, preprocessing, variance evaluation, and classification.
- **High Retention & Accuracy**:
  - **94.40%** variance retained using 100 components.
  - **94.00%** classification accuracy (94 / 100 test images correctly recognized).
- **Standalone Interactive UI (`index.html`)**:
  - **Page 1: Code & Modules**: Complete Python source code viewer, module breakdown (NumPy, scikit-learn, Matplotlib), and a 9-stage code workflow.
  - **Page 2: Interactive Results**: Animated recognition demonstration, candidate distance rankings, full face & Eigenface gallery, and interactive Euclidean distance calculator.
  - **Page 3: Mathematical Theory**: Core equations, 15 theoretical sections, and 10 expandable viva-voce questions with model answers.
- **Zero External Dependencies for UI**: Completely self-contained web platform with organized local assets that runs offline in any browser.

---

## 🛠️ Technologies Used

- **Python 3.x**
  - `numpy`: Matrix algebra, mean centering, Euclidean distance calculation.
  - `scikit-learn`: `fetch_olivetti_faces`, `train_test_split`, `PCA`, `accuracy_score`, `classification_report`.
  - `matplotlib`: Visualization of Eigenfaces, cumulative explained variance curves, and face match grids.
- **Frontend / Laboratory Interface**
  - **HTML5 & CSS3**: Modular stylesheets (`styles.css`), dark academic theme (Dark Navy / Indigo / Purple), glassmorphism, responsive grids.
  - **Vanilla JavaScript (ES6+)**: Modular script (`script.js`), multi-page tab routing, animated Euclidean distance calculations, modal lightboxes, and accordions.

---

## 📁 Project Structure

```text
Face-Recognition-EigenFace/
├── index.html                  # Interactive 3-page web dashboard & laboratory UI
├── styles.css                  # UI styling & dark academic laboratory theme
├── script.js                   # Interactive routing, animations & distance calculator
├── main.py                     # Complete 192-line Python PCA & Eigenfaces script
├── README.md                   # Comprehensive project documentation
├── STUDENT_DETAILS.md          # Team member and academic project details
├── .gitignore                  # Git exclusions for Python and system files
├── LICENSE                     # MIT License
├── assets/
│   └── images/                 # Modular face and Eigenface image assets
└── outputs/
    ├── eigenfaces.png          # Mean face + top 9 principal component Eigenfaces
    ├── explained_variance.png  # Cumulative explained variance curve (1 to 100 components)
    ├── face_matches.png        # 3x4 grid comparing test faces to closest training matches
    └── results.txt             # Verified numerical summary of the experiment
```

---

## 📊 Experimental Results

| Metric | Experimental Value |
| :--- | :--- |
| **Dataset** | AT&T Cambridge Olivetti Faces (`fetch_olivetti_faces`) |
| **Total Images** | 400 faces ($64 \times 64$ grayscale pixels) |
| **Subjects / Classes** | 40 subjects (10 images per subject) |
| **Training Set ($X_{train}$)** | 300 images (75%) |
| **Testing Set ($X_{test}$)** | 100 images (25%, stratified) |
| **Principal Components ($K$)** | 100 components |
| **Original Dimensions** | 4,096 features |
| **Variance Retained** | **94.40%** |
| **Recognition Accuracy** | **94.00%** (94 / 100 correct classifications) |

---

## 🚀 Installation & Setup

### Prerequisites

Ensure Python 3.8 or higher is installed:

```bash
python --version
```

### Install Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

---

## 💻 Execution Instructions

### 1. Run the Python Pipeline

Execute `main.py` to download/load the Olivetti dataset, compute PCA, evaluate accuracy, and output plots:

```bash
python main.py
```

Generated outputs will be saved directly into the `outputs/` folder:
- `outputs/eigenfaces.png`
- `outputs/explained_variance.png`
- `outputs/face_matches.png`
- `outputs/results.txt`

### 2. Launch the Interactive Web Dashboard

Open `index.html` directly in any web browser:

- **Windows**:
  ```powershell
  start index.html
  ```
- **macOS**:
  ```bash
  open index.html
  ```
- **Linux**:
  ```bash
  xdg-open index.html
  ```

Or serve via Python's built-in HTTP server:
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in your browser.

---

## 👥 Student & Team Information

Please refer to [`STUDENT_DETAILS.md`](STUDENT_DETAILS.md) for full member information and registration numbers.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
