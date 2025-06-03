import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

st.title("🤖 Hasil Clustering Paket Wisata")

df = pd.read_csv("package_tourism.csv")
df['combined'] = df[['Place_Tourism1','Place_Tourism2','Place_Tourism3','Place_Tourism4','Place_Tourism5']].fillna('').agg(' '.join, axis=1)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined'])

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

st.write("### Jumlah Paket per Cluster")
st.bar_chart(df['Cluster'].value_counts())

st.write("### Contoh Paket per Cluster")
for i in range(3):
    st.subheader(f"Cluster {i}")
    st.dataframe(df[df['Cluster'] == i][['City', 'Place_Tourism1', 'Place_Tourism2', 'Place_Tourism3']].head())

