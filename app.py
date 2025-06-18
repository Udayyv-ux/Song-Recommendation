import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
@st.cache_data
def load_data():
    df = pd.read_csv('Song_Recommendation_Dataset.csv')
    df = df.dropna().reset_index(drop=True)
    return df
df = load_data()
st.title('Song Recommendation System')
st.write("Dataset Columns:", df.columns.tolist())
method = st.sidebar.selectbox("Select Recommendation Method", ["Cosine Similarity (Approx)", "KNN"])
song_column = st.sidebar.selectbox("Select the column containing song names", df.select_dtypes(include=['object']).columns)
selected_song = st.selectbox("Select a song to get recommendations", df[song_column].unique())
numeric_columns = df.select_dtypes(include=['number']).columns
features = st.sidebar.multiselect("Select Features for Recommendation", numeric_columns, default=numeric_columns[:5])
X = df[features]
X_sparse = csr_matrix(X)
if method == "Cosine Similarity (Approx)":
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn.fit(X_sparse)
    song_index = df[df[song_column] == selected_song].index[0]
    distances, indices = knn.kneighbors(X_sparse[song_index])
    recommended_songs = [df.iloc[i][song_column] for i in indices[0][1:]]
elif method == "KNN":
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(X)
    song_index = df[df[song_column] == selected_song].index[0]
    distances, indices = knn.kneighbors([X.iloc[song_index]])
    recommended_songs = [df.iloc[i][song_column] for i in indices[0][1:]]
st.write("Recommended Songs:")
st.write(recommended_songs)
st.sidebar.header("Data Visualization")
plot_type = st.sidebar.selectbox("Select Plot Type", ["Scatter Plot", "Histogram"])
x_col = st.sidebar.selectbox("X-axis", numeric_columns)
y_col = st.sidebar.selectbox("Y-axis", numeric_columns) if plot_type == "Scatter Plot" else None
if plot_type == "Scatter Plot":
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    st.pyplot(fig)
elif plot_type == "Histogram":
    fig, ax = plt.subplots()
    sns.histplot(df[x_col], kde=True, ax=ax)
    st.pyplot(fig)