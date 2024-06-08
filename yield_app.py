import streamlit as st
from streamlit_shap import st_shap
import shap
import joblib
from catboost import CatBoostRegressor
import numpy as np

model_path = 'Model/CatBoostRegressor_model.joblib'
model = joblib.load(model_path)

st.title('Blueberry Crop Yield Prediction')

clone_options = [10, 12.5, 20, 25. , 37.5, 40. ]
temp_options = [ 77.4, 69.7, 86., 89. , 94.6]
rain_options = [1. , 3.77, 16. , 24. , 34.]

with st.form('Prediction Form'):
    st.header('Set the bee density and weather conditions:')
    clonesize = st.selectbox ('Area of clone in m2: ', options = clone_options )
    honeybee = st.slider('Density of honeybees/m2/min: ',min_value = 0.0,max_value = 18.4, step = 0.1, format = '%f')
    bumbles = st.slider('Density of bumblebees/m2/min: ',min_value = 0.00,max_value = 0.59, step = 0.01, format = '%f')
    andrena = st.slider('Density of andrena bees/m2/min: ',min_value = 0.00,max_value = 0.75, step = 0.01, format = '%f')
    
    osmia = st.slider('Density of osmia bees/m2/min: ',min_value = 0.00,max_value = 0.75, step = 0.01, format = '%f')
    MaxOfUpperTRange = st.selectbox('Highest Temperature Limit: ',options = temp_options)
    RainingDays = st.selectbox('Total no. of rained days: ', options = rain_options)
        
    submit_values = st.form_submit_button ('Predict')
    
if submit_values:
    data = np.array([clonesize, honeybee,bumbles, andrena,osmia, MaxOfUpperTRange,RainingDays]).reshape(1,-1)


    prediction = model.predict(data)
    
    st.header('Here is the predicted yield')
    st.success(f'{prediction[0]}')
    st.balloons()
    
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    st_shap(shap.plots.force(shap_values))

    st_shap(shap.plots.waterfall(shap_values[0]))