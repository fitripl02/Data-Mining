import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

st.title("📌 Prediksi Cluster untuk Paket Baru")

# Load dan latih ulang model
df = pd.read_csv("package_tourism.csv")
df['combined'] = df[['Place_Tourism1','Place_Tourism2','Place_Tourism3','Place_Tourism4','Place_Tourism5']].fillna('').agg(' '.join, axis=1)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined'])
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

# Formulir Input
place1 = st.text_input("Tempat Wisata 1")
place2 = st.text_input("Tempat Wisata 2")
place3 = st.text_input("Tempat Wisata 3")
place4 = st.text_input("Tempat Wisata 4")
place5 = st.text_input("Tempat Wisata 5")

if st.button("Prediksi Cluster"):
    combined_input = ' '.join(filter(None, [place1, place2, place3, place4, place5]))
    input_vector = vectorizer.transform([combined_input])
    pred = kmeans.predict(input_vector)[0]
    st.success(f"Paket ini diprediksi masuk ke dalam Cluster {pred}")

