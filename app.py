import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

st.set_page_config(
    page_title="SVD MNIST - Projeção Ur e Reconstrução",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_mnist_data(n_samples=500):
    """
    Carrega um subconjunto das imagens do conjunto de dados MNIST de 28x28 pixéis
    e normaliza os valores dos pixéis para o intervalo [0, 1].
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data[:n_samples].astype(np.float64) / 255.0
    y = mnist.target[:n_samples].astype(int)
    return X, y

@st.cache_data
def compute_svd(X_data):
    """
    Calcula a SVD na matriz de dados centrada pela média.
    Retorna a média, os dados centrados e as matrizes U, S e Vt.
    """
    X_mean = np.mean(X_data, axis=0)
    X_centered = X_data - X_mean
    # Decomposição SVD: X_centered = U * S * Vt
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_mean, X_centered, U, s, Vt

st.sidebar.title("🎛️ Parâmetros do Modelo")

n_samples = st.sidebar.slider(
    "Número de Amostras ($N$)",
    min_value=100,
    max_value=1000,
    value=300,
    step=50,
    help="Define quantas imagens do conjunto MNIST serão processadas."
)

with st.spinner("A carregar dados do MNIST e a calcular SVD..."):
    X, y = load_mnist_data(n_samples)
    X_mean, X_centered, U, s, Vt = compute_svd(X)

N, d = X.shape
max_r = min(N, d)

r = st.sidebar.slider(
    "Componentes Retidas ($r$)",
    min_value=1,
    max_value=max_r,
    value=15,
    step=1,
    help="Número de valores singulares e vetores a manter para a aproximação de posto reduzido."
)

num_display = st.sidebar.slider(
    "Dígitos a Comparar na Reconstrução",
    min_value=2,
    max_value=12,
    value=6,
    step=2
)

# Matrizes truncadas
U_r = U[:, :r]                   # N x r
s_r = s[:r]                       # r
Vt_r = Vt[:r, :]                 # r x d
Sigma_r = np.diag(s_r)           # r x r

# Reconstrução da matriz: X_hat = Mean + U_r * Sigma_r * Vt_r
X_hat_centered = np.dot(U_r, np.dot(Sigma_r, Vt_r))
X_hat = X_hat_centered + X_mean

st.title("🔬 Decomposição em Valores Singulares (SVD)")
st.subheader("Visualização da Projeção com $\mathbf{U}_r$ e Reconstrução de Imagens")
st.markdown("""
Esta aplicação permite explorar o impacto da redução de dimensionalidade por **SVD** no conjunto de dados MNIST ($28 \\times 28$ pixéis).
Analise a estrutura latente através da matriz $\mathbf{U}_r$ e veja a qualidade da reconstrução $\mathbf{\hat{X}}$ ajustando o número de componentes $r$.
""")

st.markdown("---")
st.header("1. Projeção no Espaço das Amostras ($\mathbf{U}_r$)")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"### Tabela da Matriz $\mathbf{{U}}_r \\in \\mathbb{{R}}^{{{N} \\times {r}}}$")
    st.caption("Cada linha contém as coordenadas da respetiva amostra no subespaço ortogonal.")
    
    cols_names = [f"u_{i+1}" for i in range(r)]
    df_U_r = pd.DataFrame(U_r, columns=cols_names)
    df_U_r.insert(0, "Dígito Real", y)
    
    st.dataframe(df_U_r.head(20), height=400, use_container_width=True)

with col2:
    st.markdown("### Projeção Latente 2D ($\mathbf{u}_1$ vs $\mathbf{u}_2$)")
    if r >= 2:
        fig_scatter, ax_scat = plt.subplots(figsize=(6, 4.5))
        scatter = ax_scat.scatter(
            U_r[:, 0], U_r[:, 1], 
            c=y, cmap="tab10", alpha=0.8, s=30, edgecolors="none"
        )
        ax_scat.set_xlabel("1ª Coluna de $\mathbf{U}_r$ ($\mathbf{u}_1$)", fontsize=10)
        ax_scat.set_ylabel("2ª Coluna de $\mathbf{U}_r$ ($\mathbf{u}_2$)", fontsize=10)
        ax_scat.set_title("Distribuição das Amostras no Espaço de $\mathbf{U}_r$", fontsize=11)
        
        cbar = plt.colorbar(scatter, ax=ax_scat)
        cbar.set_label("Dígito MNIST", fontsize=10)
        ax_scat.grid(True, linestyle="--", alpha=0.4)
        
        st.pyplot(fig_scatter)
    else:
        st.info("Aumente $r \\ge 2$ na barra lateral para ativar o gráfico de dispersão 2D.")

st.markdown("---")
st.header("2. Resultados da Reconstrução ($\mathbf{\hat{X}} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$)")

# Cálculo de métricas
mse_err = np.mean((X - X_hat) ** 2)
var_ret = (np.sum(s_r**2) / np.sum(s**2)) * 100

m1, m2, m3 = st.columns(3)
m1.metric("Posto de Truncagem ($r$)", f"{r}")
m2.metric("Erro Quadrático Médio (MSE)", f"{mse_err:.5f}")
m3.metric("Variância Explicada Retida", f"{var_ret:.2f}%")

st.markdown("### Comparação Visual Lado a Lado: Imagens Originais vs. Reconstruídas")

fig_rec, axes_rec = plt.subplots(2, num_display, figsize=(num_display * 2.2, 4.5))

for i in range(num_display):
    # Imagem Original
    axes_rec[0, i].imshow(X[i].reshape(28, 28), cmap="gray")
    axes_rec[0, i].set_title(f"Original: {y[i]}", fontsize=9, fontweight="bold")
    axes_rec[0, i].axis("off")

    # Imagem Reconstruída
    rec_img = np.clip(X_hat[i].reshape(28, 28), 0, 1)
    axes_rec[1, i].imshow(rec_img, cmap="gray")
    axes_rec[1, i].set_title(f"Reconst. ($r={r}$)", fontsize=9)
    axes_rec[1, i].axis("off")

plt.tight_layout()
st.pyplot(fig_rec)
