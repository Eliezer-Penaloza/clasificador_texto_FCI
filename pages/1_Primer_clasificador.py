import streamlit as st
#import pickle
import pandas as pd
#import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('stopwords')
#nltk.download('wordnet')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import altair as alt
from joblib import load



df = pd.read_csv('comunicaciones_09_07_2024.csv')
df['fecha_creacion'] = pd.to_datetime(df['fecha_creacion'])


cat = [ 'Proceso De Valoración Del Proyecto',
       'Cambio De Vocero(A) Responsable',
       'Problemas Con El Rif O Código Situr', 'Reporte De Insumos']

def limpieza_texto(texto):

    r = texto.lower()

    r = r.replace(',', ' ')

    r = r.replace(';', ' ')

    r = r.replace('.', ' ')

    r =  r.replace('á', 'a')

    r =  r.replace('é', 'e')

    r =  r.replace('í', 'i')

    r =  r.replace('ó', 'o')

    r =  r.replace('ú', 'u')

    r =  r.replace('ñ', 'n')

    r = re.sub('[^a-zA-Z]', ' ', r)


    r = r.split()

    #r = [word for word in r if word not in stopwords.words('spanish')]

    r = [word for word in r if word not in l]

    #r = [word for word in r if word not in l2]

    #corpus2.extend(r)

    #r = [lemmatizer.lemmatize(word) for word in r]

    r = ' '.join(r)

    return r
col1,col2,col3 = st.columns(3)
l = stopwords.words('spanish').copy()
l.extend(['buenas', 'saludo', 'tardes', 'revolucionario', 'mas', 'nicolas', 'chavez'])

df['categoria2'] = df['categoria'].apply(lambda x: x  if x in cat else 'categoria0')
df2 = df[df['fecha_creacion'] >= pd.to_datetime('2024-03-05')]
df2['mensaje'] = df2['mensaje'].apply(lambda x: limpieza_texto(x))
df2['long_mensaje_2'] = df2['mensaje'].apply(lambda x: len(x.split()))
df2 = df2[df2['long_mensaje_2'] >= 5]
df2 = df2.drop_duplicates()

with open("vectorizer_categoria_11072024(1).joblib", "rb") as file:
        loaded_tokenizer = load(file)
        

with open("modelo_categoria_11072024(1).joblib", "rb") as file:
        loaded_model = load(file)

df2['categoria_modelo'] = df2['mensaje'].apply(lambda x : loaded_model.predict(loaded_tokenizer.transform([x]))[0])
#df2['categoria_2'] = df2['categoria'].apply(lambda x: x  if x in cat else 'categoria0')





st.header('La cantidad de categorias son: ' + str(len(df['categoria'].drop_duplicates().to_list())))

st.write('Sin embargo, el problema se simplificó en un clasificador de 5 categorías. Resultando las siguientes proporciones.')



t = df['categoria2'].value_counts()
auxt = pd.DataFrame({'Categoria':t.index, 'cantidad': t.values})

auxt['porcentaje'] = auxt['cantidad'] / auxt['cantidad'].sum() * 100

base_chart = alt.Chart(auxt, title = 'Proporción de las categorias').mark_arc().encode(
    theta='cantidad',
    color='Categoria',
    tooltip=['Categoria', 'cantidad', 'porcentaje']  # Add tooltips to show values on hover
).properties(center=True,height = 450, width = 600)


st.write(' ')
st.altair_chart(base_chart)
st.write(' ')
st.header('Y como resultado tenemos la siguiente tabla de rendimientos.')
st.write(' ')
df3 = pd.DataFrame(classification_report(df2['categoria2'], df2['categoria_modelo'], digits=4, output_dict= True)).transpose()
st.dataframe(df3)
