import streamlit as st
#import pickle
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
#nltk.download('stopwords')
#nltk.download('wordnet')
#from sklearn.linear_model import LogisticRegression
import re
from joblib import dump
from joblib import load



l = ['de',
 'la',
 'que',
 'el',
 'en',
 'y',
 'a',
 'los',
 'del',
 'se',
 'las',
 'por',
 'un',
 'para',
 'con',
 'no',
 'una',
 'su',
 'al',
 'lo',
 'como',
 'más',
 'pero',
 'sus',
 'le',
 'ya',
 'o',
 'este',
 'sí',
 'porque',
 'esta',
 'entre',
 'cuando',
 'muy',
 'sin',
 'sobre',
 'también',
 'me',
 'hasta',
 'hay',
 'donde',
 'quien',
 'desde',
 'todo',
 'nos',
 'durante',
 'todos',
 'uno',
 'les',
 'ni',
 'contra',
 'otros',
 'ese',
 'eso',
 'ante',
 'ellos',
 'e',
 'esto',
 'mí',
 'antes',
 'algunos',
 'qué',
 'unos',
 'yo',
 'otro',
 'otras',
 'otra',
 'él',
 'tanto',
 'esa',
 'estos',
 'mucho',
 'quienes',
 'nada',
 'muchos',
 'cual',
 'poco',
 'ella',
 'estar',
 'estas',
 'algunas',
 'algo',
 'nosotros',
 'mi',
 'mis',
 'tú',
 'te',
 'ti',
 'tu',
 'tus',
 'ellas',
 'nosotras',
 'vosotros',
 'vosotras',
 'os',
 'mío',
 'mía',
 'míos',
 'mías',
 'tuyo',
 'tuya',
 'tuyos',
 'tuyas',
 'suyo',
 'suya',
 'suyos',
 'suyas',
 'nuestro',
 'nuestra',
 'nuestros',
 'nuestras',
 'vuestro',
 'vuestra',
 'vuestros',
 'vuestras',
 'esos',
 'esas',
 'estoy',
 'estás',
 'está',
 'estamos',
 'estáis',
 'están',
 'esté',
 'estés',
 'estemos',
 'estéis',
 'estén',
 'estaré',
 'estarás',
 'estará',
 'estaremos',
 'estaréis',
 'estarán',
 'estaría',
 'estarías',
 'estaríamos',
 'estaríais',
 'estarían',
 'estaba',
 'estabas',
 'estábamos',
 'estabais',
 'estaban',
 'estuve',
 'estuviste',
 'estuvo',
 'estuvimos',
 'estuvisteis',
 'estuvieron',
 'estuviera',
 'estuvieras',
 'estuviéramos',
 'estuvierais',
 'estuvieran',
 'estuviese',
 'estuvieses',
 'estuviésemos',
 'estuvieseis',
 'estuviesen',
 'estando',
 'estado',
 'estada',
 'estados',
 'estadas',
 'estad',
 'he',
 'has',
 'ha',
 'hemos',
 'habéis',
 'han',
 'haya',
 'hayas',
 'hayamos',
 'hayáis',
 'hayan',
 'habré',
 'habrás',
 'habrá',
 'habremos',
 'habréis',
 'habrán',
 'habría',
 'habrías',
 'habríamos',
 'habríais',
 'habrían',
 'había',
 'habías',
 'habíamos',
 'habíais',
 'habían',
 'hube',
 'hubiste',
 'hubo',
 'hubimos',
 'hubisteis',
 'hubieron',
 'hubiera',
 'hubieras',
 'hubiéramos',
 'hubierais',
 'hubieran',
 'hubiese',
 'hubieses',
 'hubiésemos',
 'hubieseis',
 'hubiesen',
 'habiendo',
 'habido',
 'habida',
 'habidos',
 'habidas',
 'soy',
 'eres',
 'es',
 'somos',
 'sois',
 'son',
 'sea',
 'seas',
 'seamos',
 'seáis',
 'sean',
 'seré',
 'serás',
 'será',
 'seremos',
 'seréis',
 'serán',
 'sería',
 'serías',
 'seríamos',
 'seríais',
 'serían',
 'era',
 'eras',
 'éramos',
 'erais',
 'eran',
 'fui',
 'fuiste',
 'fue',
 'fuimos',
 'fuisteis',
 'fueron',
 'fuera',
 'fueras',
 'fuéramos',
 'fuerais',
 'fueran',
 'fuese',
 'fueses',
 'fuésemos',
 'fueseis',
 'fuesen',
 'sintiendo',
 'sentido',
 'sentida',
 'sentidos',
 'sentidas',
 'siente',
 'sentid',
 'tengo',
 'tienes',
 'tiene',
 'tenemos',
 'tenéis',
 'tienen',
 'tenga',
 'tengas',
 'tengamos',
 'tengáis',
 'tengan',
 'tendré',
 'tendrás',
 'tendrá',
 'tendremos',
 'tendréis',
 'tendrán',
 'tendría',
 'tendrías',
 'tendríamos',
 'tendríais',
 'tendrían',
 'tenía',
 'tenías',
 'teníamos',
 'teníais',
 'tenían',
 'tuve',
 'tuviste',
 'tuvo',
 'tuvimos',
 'tuvisteis',
 'tuvieron',
 'tuviera',
 'tuvieras',
 'tuviéramos',
 'tuvierais',
 'tuvieran',
 'tuviese',
 'tuvieses',
 'tuviésemos',
 'tuvieseis',
 'tuviesen',
 'teniendo',
 'tenido',
 'tenida',
 'tenidos',
 'tenidas',
 'tened',
 'buenas',
 'saludo',
 'tardes',
 'revolucionario',
 'mas',
 'nicolas',
 'chavez']




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

#l = stopwords.words('spanish').copy()
l.extend(['buenas', 'saludo', 'tardes', 'revolucionario', 'mas', 'nicolas', 'chavez'])

st.set_page_config(layout="wide")

col1,col2,col3,col4,col5 = st.columns(5)

img1 = 'LOGOHORIZONTAL.jpeg'#'/home/epenaloza/Descargas/LOGOHORIZONTAL.jpeg'
img2 = '-5140927617266986116_121.jpg'#'EPT/compartir/clasificador_texto/-5140927617266986116_121.jpg'

with col1:
    st.image(img1)
with col5:
    st.image(img2)

with open('config.yml') as file:
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

    #with open("primer_modelo(1).pkl", "rb") as file:
    loaded_model = load('modelo_categoria.joblib')

    #with open("count_vectorizer (1).pkl", "rb") as file:
    loaded_tokenizer = load('vectorizer_categoria.joblib')

    #with open("primer_modelo_responsable.pkl", "rb") as file:
    loaded_model_responsable = load('modelo_responsable.joblib')

    #with open("count_vectorizer_responsable.pkl", "rb") as file:
    loaded_tokenizer_responsable = load('vectorizer_responsable.joblib')     

    txt = st.text_input('Por favor, ingrese un texto')


if txt != '':

    st.write('El texto pertenece a la categoria: ' + loaded_model.predict(loaded_tokenizer.transform([limpieza_texto(txt)])[0]))
    st.write('Debe ser atendido por: ' + loaded_model_responsable.predict(loaded_tokenizer_responsable.transform([limpieza_texto(txt)])[0]))  
else:
    st.write('El texto pertenece a la categoria: ' )
    st.write('Debe ser atendido por: ' ) 
        
        
    authenticator.logout('Cerrar sesión', 'main', key= 'unique_key')
