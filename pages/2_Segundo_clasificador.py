import streamlit as st
import pickle
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
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import altair as alt
from joblib import load



def cfg(x):
   l_cfg = ['Consejo Federal de Gobierno - FCI-Sofia Margarita Fleury Hernandez',
 'Consejo Federal de Gobierno - FCI-Cruz David Mata Noguera',
 'Consejo Federal de Gobierno - FCI-Ricardo Jose Musett Roman',
 'Consejo Federal de Gobierno - FCI-Luisana Valentina Velasquez Fermin',
 'Consejo Federal de Gobierno - FCI-Sin Responsable',
 'Consejo Federal de Gobierno - FCI-Joali Gabriela Moreno Pinto',
 'Consejo Federal de Gobierno - FCI-Erika Victoria Mertz Rodriguez',
 'Consejo Federal de Gobierno - FCI-Adriana Teresa Hurtado Hernandez']

   l_com = [ 'Ministerio del Poder Popular para las Comunas y los Movimientos Sociales-Cesar Eduardo Carrero Aristizabal',
            'Ministerio del Poder Popular para las Comunas y los Movimientos Sociales-Sin Responsable',
 'Ministerio del Poder Popular para las Comunas y los Movimientos Sociales-Cesar Will Martinez Infante',
 'Ministerio del Poder Popular para las Comunas y los Movimientos Sociales-Hernan Jose Vargas Perez',
 'Ministerio del Poder Popular para las Comunas y los Movimientos Sociales-Fernando Jose Rodriguez Meza',
 'Ministerio del Poder Popular para las Comunas y los Movimientos Sociales-Ciro Antonio Rodriguez Villanueva']

   l_mer = ['Gobernación del estado Bolivariano de Mérida-Sin Responsable']

   l_gua = ['Gobernación del Estado Bolivariano de Guárico-Greidymar Nohelia Barrios',
            'Gobernación del Estado Bolivariano de Guárico-Sin Responsable',
            'Gobernación del Estado Bolivariano de Guárico-Yolkis Yoliana Villafranca Ovalles']

   if x  in l_cfg:
      return "OAC"
   elif x == 'Consejo Federal de Gobierno - FCI-Luis German Rivas Zambrano':
      return 'SYEPP'
   elif x == 'Consejo Federal de Gobierno - FCI-Ciro Antonio Rodriguez Villanueva':
      return 'Territorial'
   elif x in l_com:
      return 'Comunas'
   elif x in l_gua:
      return 'Gob_Guarico'
   else:
      return 'Gob_Merida'


df = pd.read_csv('/home/epenaloza/Documentos/trabajos_python/entorno_virtual/venv2/next/comunicaciones_09_07_2024.csv')
df['fecha_creacion'] = pd.to_datetime(df['fecha_creacion'])
df['nueva_etiqueta'] = df['receptor'] +'-' + df['responsable']
df['responsable2'] = df['nueva_etiqueta'].apply(lambda x: cfg(x))

cat = ['Gob_Guarico', 'OAC']

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

df['categoria2'] = df['responsable2'].apply(lambda x: x  if x in cat else 'categoria0')
df2 = df[df['fecha_creacion'] >= pd.to_datetime('2024-03-05')]
df2['mensaje'] = df2['mensaje'].apply(lambda x: limpieza_texto(x))
df2['long_mensaje_2'] = df2['mensaje'].apply(lambda x: len(x.split()))
df2 = df2[df2['long_mensaje_2'] >= 5]
df2 = df2.drop_duplicates()

with open("/home/epenaloza/Documentos/trabajos_python/entorno_virtual/venv2/EPT/compartir/clasificador_texto/vectorizer_responsable.joblib", "rb") as file:
        loaded_tokenizer = load(file)
        

with open("/home/epenaloza/Documentos/trabajos_python/entorno_virtual/venv2/EPT/compartir/clasificador_texto/modelo_responsable.joblib", "rb") as file:
        loaded_model = load(file)

df2['categoria_modelo'] = df2['mensaje'].apply(lambda x : loaded_model.predict(loaded_tokenizer.transform([x]))[0])
#df2['categoria_2'] = df2['categoria'].apply(lambda x: x  if x in cat else 'categoria0')





st.header('La cantidad de categorias son: ' + str(len(df['responsable2'].drop_duplicates().to_list())))

st.write('Sin embargo, el problema se simplificó en un clasificador de 3 categorías. Resultando las siguientes proporciones.')



t = df['responsable2'].value_counts()
auxt = pd.DataFrame({'Categoria':t.index, 'cantidad': t.values})

auxt['porcentaje'] = auxt['cantidad'] / auxt['cantidad'].sum() * 100

base_chart = alt.Chart(auxt, title = 'Proporción de las categorias').mark_arc().encode(
    theta='cantidad',
    color='Categoria',
    tooltip=['Categoria', 'cantidad', 'porcentaje']  # Add tooltips to show values on hover
)


st.write(' ')
st.altair_chart(base_chart)
st.write(' ')
st.header('Y como resultado tenemos la siguiente tabla de rendimientos.')
st.write(' ')
df3 = pd.DataFrame(classification_report(df2['categoria2'], df2['categoria_modelo'], digits=4, output_dict= True)).transpose()
st.dataframe(df3)
