import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 K-Means Clustering & Elbow Method Dashboard")
st.markdown("""
This app performs automated **K-Means Clustering** on your dataset, maps out the **Elbow Method** to find the optimal number of clusters, 
and projects the data into an interactive **3D PCA space**.
""")

# --- FILE LOADING ---
# Looks for the file in the root GitHub directory, or falls back to an uploader
@st.cache_data
def load_data(file_path="iris.xlsx - iris.csv"):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

df = load_data()

# Sidebar fallback if file isn't committed to GitHub yet
if df is None:
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload 'iris.xlsx - iris.csv'", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("👋 Please commit 'iris.xlsx - iris.csv' to your GitHub repository or upload it via the sidebar to begin.")
        st.stop()

# --- FEATURE SELECTION & PREPROCESSING ---
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
X = df[features]

# Scale features for distance-based calculation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Cluster Settings")
max_k = st.sidebar.slider("Max K for Elbow Method", min_value=5, max_value=15, value=10)

# --- SPLIT LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 1. The Elbow Method")
    st.write("Look for the 'inflection point' or elbow where the WCSS drop slows down.")
    
    # Calculate WCSS
    wcss = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot Elbow Curve
    fig_elbow, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(k_range, wcss, 'bx-', color='#1f77b4', linewidth=2)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('WCSS (Inertia)')
    ax.set_title('Determining Optimal K')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_elbow)

# Dynamic K-selection based on Elbow analysis
optimal_k = st.sidebar.number_input("Select Target Clusters (k)", min_value=1, max_value=max_k, value=3)

# Run final K-Means algorithm
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype(str) # Convert to string category for discrete coloring

with col2:
    st.subheader("🧊 2. Dynamic 3D PCA Projection")
    st.write(f"Visualizing data points grouped into **{optimal_k} clusters**.")
    
    # Dimensionality reduction for 3D mapping
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    df['PCA3'] = X_pca[:, 2]
    
    # Render interactive Plotly Graph
    fig_3d = px.scatter_3d(
        df, 
        x='PCA1', y='PCA2', z='PCA3',
        color='Cluster',
        labels={'PCA1': 'PC 1', 'PCA2': 'PC 2', 'PCA3': 'PC 3'},
        hover_data=features,
        opacity=0.85,
        color_discrete_sequence=px.colors.qualitative.D3
    )
    
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        ),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05)
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# --- DATA VIEW SECTION ---
st.markdown("---")
st.subheader("📋 Clustered Dataset Preview")
st.dataframe(df[features + ['Cluster']].head(20), use_container_width=True)