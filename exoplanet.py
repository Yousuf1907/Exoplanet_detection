import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('exoplanet_detection_model')

st.set_page_config(
    page_title="Astronomy ML App",
    page_icon="🌟",
    # initial_sidebar_state="expanded",  # Sidebar is expanded by default
)

def main():
    st.title('Exoplanet Detection App')
    st.write('This app predicts exoplanet detection based on input features.')

    st.header('Input Features')
    age = st.slider('Star Temperature (K)', min_value=2000, max_value=10000)
    radius = st.slider('Star Radius (R⊙)', min_value=0.1, max_value=10.0)
    luminosity = st.slider('Star Luminosity (L⊙)', min_value=0.001, max_value=10000.0)
    exoplanet_radius = st.slider('Exoplanet Radius (R⊕)', min_value=0.1, max_value=5.0)
    exoplanet_distance = st.slider('Exoplanet Distance from Star (AU)', min_value=0.01, max_value=100.0)
    exoplanet_orbit = st.slider('Exoplanet Orbital Period (days)', min_value=0.1, max_value=1000.0)
    exoplanet_mass = st.slider('Exoplanet Mass (M⊕)', min_value=0.01, max_value=10.0)

    if st.button('Predict'):
        user_input = np.array([age, radius, luminosity, exoplanet_radius, exoplanet_distance, exoplanet_orbit, exoplanet_mass])
        user_input = user_input.reshape(1, -1)
        
        prediction = model.predict(user_input)

        if prediction[0][0] > 0.5:
            st.write('Prediction: Exoplanet Detected')
        else:
            st.write('Prediction: No Exoplanet Detected')

if __name__ == '__main__':
    main()
