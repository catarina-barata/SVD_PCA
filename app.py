import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

st.set_page_config(
    page_title="SVD vs PCA Demo — MNIST",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. Data Loading and SVD Computation
# -----------------------------------------------------------------------------
@st.cache_data
def load_mnist_data(n_samples=500):
    """Loads a subset of the MNIST dataset."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data[:n_samples].astype(np.float64) / 255.0  # Normalize to [0, 1]
    y = mnist.target[:n_samples].astype(int)
    return X, y

@st.cache_data
def compute_svd(X_data):
    """Computes mean-centered SVD."""
    X_mean = np.mean(X_data, axis=0)
    X_centered = X_data - X_mean
    # Full Economy SVD: X_c = U * S * Vt
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_mean, X_centered, U, s, Vt

# -----------------------------------------------------------------------------
# 2. Main Interface & Sidebar Controls
# -----------------------------------------------------------------------------
st.title("SVD and PCA Equivalency & Analysis — MNIST")
st.markdown("""
This interactive demo explores the relationship between **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**:
- **PCA Scores via $V_r$:** $\mathbf{Z} = (\mathbf{X} - \boldsymbol{\mu}) \mathbf{V}_r$
- **PCA Scores via SVD:** $\mathbf{Z} = \mathbf{U}_r \mathbf{\Sigma}_r$
- **Variance relación:** $\lambda_i = \frac{\sigma_i^2}{N - 1}$
""")

st.sidebar.header("Dataset & Truncation Parameters")
n_samples = st.sidebar.slider("Number of Samples ($N$)", min_value=100, max_value=1000, value=400, step=50)

with st.spinner("Loading MNIST data and computing SVD..."):
    X, y = load_mnist_data(n_samples)
    X_mean, X_centered, U, s, Vt = compute_svd(X)

N, d = X.shape
max_r = min(N, d)

r = st.sidebar.slider("Retained Components ($r$)", min_value=1, max_value=max_r, value=20, step=1)

# -----------------------------------------------------------------------------
# Section 1: Evolution of Explained Variance
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("1. Evolution of Explained Variance")

# Variance derived from Singular Values: lambda_i = s_i^2 / (N - 1)
eigenvalues = (s ** 2) / (N - 1)
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

col_metric1, col_metric2, col_metric3 = st.columns(3)
col_metric1.metric("Retained Components ($r$)", f"{r}")
col_metric2.metric("Variance Retained by $r$ Components", f"{cumulative_variance[r-1] * 100:.2f}%")
col_metric3.metric("Total Eigenvalue Variance", f"{total_variance:.4f}")

fig_var, ax_var = plt.subplots(figsize=(10, 3.5))
ax_var.plot(range(1, max_r + 1), cumulative_variance * 100, label="Cumulative Explained Variance (%)", color="navy", linewidth=2)
ax_var.bar(range(1, max_r + 1), explained_variance_ratio * 100, label="Individual Component Variance (%)", color="skyblue", alpha=0.6)

# Highlight current threshold r
ax_var.axvline(x=r, color="red", linestyle="--", label=f"Selected $r={r}$")
ax_var.axhline(y=cumulative_variance[r-1] * 100, color="red", linestyle=":", alpha=0.7)

ax_var.set_xlabel("Principal Component Index ($i$)")
ax_var.set_ylabel("Explained Variance (%)")
ax_var.set_title("Scree Plot and Cumulative Explained Variance Ratio")
ax_var.legend(loc="center right")
ax_var.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig_var)

# -----------------------------------------------------------------------------
# Section 2: Projections Equivalence: X_c * V_r vs. U_r * Sigma_r
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("2. Projection Equivalence: $(\mathbf{X} - \boldsymbol{\mu})\mathbf{V}_r$  vs.  $\mathbf{U}_r \mathbf{\Sigma}_r$")

st.markdown("""
In PCA, the $N \times r$ latent coordinates (**Principal Component Scores**) can be computed in two mathematically identical ways:
1. **Via Right Singular Vectors ($V_r$):** $\mathbf{Z}_{\text{features}} = \mathbf{X}_c \mathbf{V}_r$
2. **Via Left Singular Vectors and Singular Values ($U_r, \Sigma_r$):** $\mathbf{Z}_{\text{samples}} = \mathbf{U}_r \mathbf{\Sigma}_r$
""")

# Truncate components
V_r = Vt[:r, :].T                 # d x r
U_r = U[:, :r]                     # N x r
Sigma_r = np.diag(s[:r])           # r x r

# Compute both projections
Z_from_V = np.dot(X_centered, V_r)     # Projection via Features: (N x d) * (d x r) -> N x r
Z_from_U = np.dot(U_r, Sigma_r)         # Projection via Samples:  (N x r) * (r x r) -> N x r

# Calculate absolute numerical difference between both projections
max_diff = np.max(np.abs(Z_from_V - Z_from_U))

st.success(f"**Maximum Absolute Difference between $(\mathbf{{X}}_c \mathbf{{V}}_r)$ and $(\mathbf{{U}}_r \mathbf{{\Sigma}}_r)$:** `{max_diff:.10e}` (Numerically Identical)")

col_proj1, col_proj2 = st.columns(2)

with col_proj1:
    st.subheader("Projection using $\mathbf{X}_c \mathbf{V}_r$")
    if r >= 2:
        fig_p1, ax_p1 = plt.subplots(figsize=(5, 4))
        sc1 = ax_p1.scatter(Z_from_V[:, 0], Z_from_V[:, 1], c=y, cmap="tab10", alpha=0.7, s=20)
        ax_p1.set_xlabel("Component 1 ($\mathbf{v}_1$)")
        ax_p1.set_ylabel("Component 2 ($\mathbf{v}_2$)")
        ax_p1.set_title("$\mathbf{Z} = \mathbf{X}_c \mathbf{V}_r$")
        plt.colorbar(sc1, ax=ax_p1, label="Digit Label")
        ax_p1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_p1)
    else:
        st.info("Select $r \ge 2$ to plot a 2D scatter plot.")

with col_proj2:
    st.subheader("Projection using $\mathbf{U}_r \mathbf{\Sigma}_r$")
    if r >= 2:
        fig_p2, ax_p2 = plt.subplots(figsize=(5, 4))
        sc2 = ax_p2.scatter(Z_from_U[:, 0], Z_from_U[:, 1], c=y, cmap="tab10", alpha=0.7, s=20)
        ax_p2.set_xlabel("Component 1 ($\mathbf{u}_1 \sigma_1$)")
        ax_p2.set_ylabel("Component 2 ($\mathbf{u}_2 \sigma_2$)")
        ax_p2.set_title("$\mathbf{Z} = \mathbf{U}_r \mathbf{\Sigma}_r$")
        plt.colorbar(sc2, ax=ax_p2, label="Digit Label")
        ax_p2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_p2)
    else:
        st.info("Select $r \ge 2$ to plot a 2D scatter plot.")

# -----------------------------------------------------------------------------
# Section 3: Image Reconstruction Comparison
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("3. Image Reconstruction using $r$ Components")

# Reconstruct low-rank approximation: X_hat = Mean + U_r * Sigma_r * V_r^T
X_reconstructed = np.dot(Z_from_U, V_r.T) + X_mean

num_display = st.slider("Digits to preview", min_value=4, max_value=12, value=6, step=2)

fig_rec, axes_rec = plt.subplots(2, num_display, figsize=(num_display * 2, 4))

for i in range(num_display):
    # Original Image
    axes_rec[0, i].imshow(X[i].reshape(28, 28), cmap="gray")
    axes_rec[0, i].set_title(f"Original ({y[i]})", fontsize=9)
    axes_rec[0, i].axis("off")

    # Reconstructed Image
    axes_rec[1, i].imshow(np.clip(X_reconstructed[i].reshape(28, 28), 0, 1), cmap="gray")
    axes_rec[1, i].set_title(f"Reconstructed ($r={r}$)", fontsize=9)
    axes_rec[1, i].axis("off")

plt.tight_layout()
st.pyplot(fig_rec)
