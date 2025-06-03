import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Halaman EDA (Exploratory Data Analysis)")
df = pd.read_csv("package_tourism.csv")

st.write("### Cuplikan Data")
st.dataframe(df.head())

st.write("### Jumlah Paket per Kota")
st.bar_chart(df['City'].value_counts())

df['Jumlah_Tujuan'] = df.loc[:, 'Place_Tourism1':'Place_Tourism5'].notna().sum(axis=1)
st.write("### Distribusi Jumlah Tujuan per Paket")
fig, ax = plt.subplots()
sns.histplot(df['Jumlah_Tujuan'], kde=True, bins=5, ax=ax)
st.pyplot(fig)

