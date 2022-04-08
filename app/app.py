import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import yaml
import pandas as pd
import pickle
import altair as alt

from PIL import Image

with open(r'app/app.yaml') as file:
        app_load = yaml.safe_load(file)

IMAGE_WIDTH = app_load['IMAGE_WIDTH']
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_DEPTH = app_load['IMAGE_DEPTH']


def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)
    

def predict_image(path, model, how):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction 
    
    Returns
    -------
    Predicted class
    """
    if how == "neuronnes" :
        images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
        prediction_vector = model.predict(images)
        
    if how == "SVM" :
        images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))).flatten()])
        prediction_vector = model.predict_proba(images)

    return prediction_vector


def load_model(path, how):
    """Load tf/Keras model for prediction
    """    
    if how == "neuronnes" :
        model = tf.keras.models.load_model('models/'+path+'.h5')
    if how == "SVM" :
        model = pickle.load(open('models/'+path+'.pkl', 'rb'))
        
    return model
    


st.title("Identification d'avion")

col1, col2= st.columns(2)

with col1:
    st.header("Choisir une image")

    uploaded_file = st.file_uploader("Charger une image d'avion")

    if uploaded_file:
        loaded_image = load_image(uploaded_file)
        st.image(loaded_image)

with col2:
    st.header("Identifier l'avion")
    
    model_select = st.radio(
     "Quel type de modèle voulez-vous utiliser ?",
     ('Réseau de neuronnes', 'SVM'))      
    
    option = st.selectbox(
     'Que voulez-vous identifier ?',
     ('','Constructeur', 'Famille'),
    index = 0,
    disabled=(uploaded_file is None))

    st.write('Vous avez sélectionné:', option)

    predict_btn = st.button("Identifier", disabled=(option == '' ))
    
    if option == 'Constructeur' :
        if model_select == 'Réseau de neuronnes' :
            model = load_model('manufacturer','neuronnes')
        if model_select == 'SVM' :
            model = load_model('manufacturer','SVM')

    if option == 'Famille' :
        if model_select == 'Réseau de neuronnes' :
            model = load_model('family','neuronnes')
        if model_select == 'SVM' :
            model = load_model('family','SVM')
        

    if predict_btn:
        
                
        if option == 'Constructeur' :
            with open(r'models/manufacturer_label.yaml') as file:
                label = yaml.safe_load(file)
                

        if option == 'Famille' :
            with open(r'models/family_label.yaml') as file:
                label = yaml.safe_load(file)    
                
                
        if model_select == 'Réseau de neuronnes' :
            prediction = predict_image(uploaded_file, model,'neuronnes')
            predicted_classes = np.argmax(prediction,axis =1)
            id_label = label[predicted_classes[0]]
            st.write(f"C'est un : {id_label}, avec une probabilité de : {round(prediction[0][predicted_classes[0]]*100,2)} %")

            
        if model_select == 'SVM' :
            prediction = predict_image(uploaded_file, model,'SVM')
            predicted_classes = np.argmax(prediction[0])
            id_label = label[predicted_classes]

            st.write(f"C'est un : {id_label}, avec une probabilité de : {round(prediction[0][predicted_classes]*100,2)} %")
            
        
        chart_data = pd.DataFrame(
                         prediction[0],
                         index=list(label.values()),
                    columns = ['proba'])
        chart_data = chart_data.sort_values(by=['proba'],ascending=False)
        chart_data = chart_data.head(10)        

        #st.bar_chart(chart_data,
        #                use_container_width=True)
        
        data = pd.melt(chart_data.reset_index(), id_vars=["index"])
        
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("value", type="quantitative", title="Probabilité"),
                y=alt.Y("index", type="nominal", title=""),
            )
         )
        st.altair_chart(chart, use_container_width=True)
        

    

