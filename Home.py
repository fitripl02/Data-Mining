import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache
def load_data():
    return pd.read_csv('data/package_tourism.csv')

df = load_data()

# Sidebar navigation
st.sidebar.title('Navigasi')
page = st.sidebar.radio('Pilih Halaman:', 
                        ['EDA', 'Clustering', 'Prediksi'])

if page == 'EDA':
    st.title('Exploratory Data Analysis')
    
    st.header('Data Preview')
    st.write(df.head())
    
    st.header('Statistik Deskriptif')
    st.write(df.describe())
    
    st.header('Distribusi Kota')
    fig, ax = plt.subplots()
    df['City'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
elif page == 'Clustering':
    st.title('Hasil Clustering Paket Wisata')
    
    # Combine all places
    places = pd.get_dummies(df.filter(regex='Place_Tourism'))
    
    # Clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(places)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    
    df['Cluster'] = clusters
    
    st.header('Cluster Distribution')
    fig, ax = plt.subplots()
    df['Cluster'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
    st.header('Data dengan Cluster')
    st.write(df[['Package', 'City', 'Cluster']])
    
else:
    st.title('Prediksi Cluster Paket Wisata')
    
    st.header('Pilih Tempat Wisata')
    
    # Get unique places
    all_places = set()
    for i in range(1,6):
        all_places.update(df[f'Place_Tourism{i}'].dropna().unique())
    
    selected = st.multiselect('Pilih tempat wisata:', sorted(all_places))
    
    if st.button('Prediksi'):
        if len(selected) == 0:
            st.warning('Pilih minimal 1 tempat wisata')
        else:
            # Create input vector
            input_data = pd.DataFrame([[1 if place in selected else 0 for place in all_places]], 
                                    columns=sorted(all_places))
            
            # Scale and predict
            scaler = StandardScaler()
            scaled = scaler.fit_transform(input_data)
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(scaled)
            cluster = kmeans.predict(scaled)[0]
            
            st.success(f'Paket wisata diprediksi masuk Cluster {cluster}')
