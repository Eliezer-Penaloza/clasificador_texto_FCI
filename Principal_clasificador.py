import streamlit as st
import pickle
import pandas as pd
#import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
#from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.linear_model import LogisticRegression
import re




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

l = stopwords.words('spanish').copy()
l.extend(['buenas', 'saludo', 'tardes', 'revolucionario', 'mas', 'nicolas', 'chavez'])

st.set_page_config(layout="wide")

col1,col2,col3,col4,col5 = st.columns(5)

img1 = 'LOGOHORIZONTAL.jpeg'#'/home/epenaloza/Descargas/LOGOHORIZONTAL.jpeg'
img2 = '-5140927617266986116_121.jpg'#'EPT/compartir/clasificador_texto/-5140927617266986116_121.jpg'

with col1:
    st.image(img1)
with col5:
    st.image(img2)

with open('clasificador_texto/config.yml') as file:
    config = yaml.load(file, Loader= SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized'],

)

authentication_status = authenticator.login(fields={'Form name': 'Ingresar:', 'Username':'Usuario:'}, location='main')

if st.session_state['authentication_status'] == False:
    st.error('Username/password is incorrect')

elif st.session_state['authentication_status'] == None:
    st.warning('Please enter your username and password')

elif st.session_state['authentication_status'] == True:



    


    st.write(f'¡Bienvenido, *{st.session_state["name"]}*!')

    st.write('Bienvenidos al clasificador de texto')

    with open("primer_modelo(1).pkl", "rb") as file:
        loaded_model = pickle.load(file)

    with open("count_vectorizer (1).pkl", "rb") as file:
        loaded_tokenizer = pickle.load(file)

    with open("primer_modelo_responsable.pkl", "rb") as file:
        loaded_model_responsable = pickle.load(file)

    with open("count_vectorizer_responsable.pkl", "rb") as file:
        loaded_tokenizer_responsable = pickle.load(file)     

    txt = st.text_input('Por favor, ingrese un texto')


if txt != '':

    st.write('El texto pertenece a la categoria: ' + loaded_model.predict(loaded_tokenizer.transform([limpieza_texto(txt)])[0]))
    st.write('Debe ser atendido por: ' + loaded_model_responsable.predict(loaded_tokenizer_responsable.transform([limpieza_texto(txt)])[0]))  
        
        
    authenticator.logout('Cerrar sesión', 'main', key= 'unique_key')
