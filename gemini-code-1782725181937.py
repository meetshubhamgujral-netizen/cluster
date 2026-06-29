import streamlit as st
import pandas as pd
import numpy as np
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
@st.cache_data
def load_data(file_path="iris.xlsx - iris.csv"):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

df = load_data()

# Fallback UI if the file isn't found in the root directory
if df is None:
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload 'iris.xlsx - iris.csv'", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("👋 Setup Tip: Commit 'iris.xlsx - iris.csv' to the root of your GitHub repository for automatic loading.")
        st.stop()

# --- FEATURE SELECTION & PREPROCESSING ---
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
X = df[features]

# Scale features for distance-based calculations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Cluster Settings")
max_k = st.sidebar.slider("Max K for Elbow Method", min_value=5, max_value=15, value=10)

# --- SPLIT LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 1. The Elbow Method")
    st.write("Identify the inflection point where the WCSS drop flattens out.")
    
    # Calculate WCSS (Within-Cluster Sum of Squares)
    wcss = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Render interactive Line Chart using Plotly
    df_elbow = pd.DataFrame({'k': list(k_range), 'WCSS': wcss})
    fig_elbow = px.line(
        df_elbow, 
        x='k', 
        y='WCSS', 
        markers=True,
        template="plotly_white"
    )
    fig_elbow.update_traces(line_color='#1f77b4', marker=dict(size=8, symbol='circle'))
    fig_elbow.update_layout(
        xaxis=dict(title='Number of clusters (k)', tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title='WCSS (Inertia)'),
        margin=dict(l=10, r=10, t=10, b=10),
        height=400
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

# Dynamic Cluster Selection Input
optimal_k = st.sidebar.number_input("Select Target Clusters (k)", min_value=1, max_value=max_k, value=3)

# Execute final K-Means 
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype(str) # Convert to discrete categorical string for explicit coloring

with col2:
    st.subheader("🧊 2. Dynamic 3D PCA Projection")
    st.write(f"Visualizing spatial distribution using **{optimal_k} clusters**.")
    
    # Dimensionality reduction for 3D layout
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    df['PCA3'] = X_pca[:, 2]
    
    # Render interactive 3D Scatter Plot
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
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
        height=400
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# --- DATAFRAME VIEW ---
st.markdown("---")
st.subheader("📋 Clustered Dataset Preview")
st.dataframe(df[features + ['Cluster']].head(20), use_container_width=True)
