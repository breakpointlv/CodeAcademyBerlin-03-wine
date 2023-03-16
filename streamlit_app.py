import pickle

import streamlit as st
import pandas as pd

initialized = False

def predict():
    with st.spinner('Tasting wine...'):
        stats = st.columns(7)

        with stats[0]:
            st.metric(label="Alcohol", value=str(alcohol) + " %")
        with stats[1]:
            st.metric(label="Suplhates", value=str(suplhates))
        with stats[2]:
            st.metric(label="pH", value=str(ph))
        with stats[3]:
            st.metric(label="Density", value=str(density))
        with stats[4]:
            st.metric(label="Total SO2", value=str(so2))
        with stats[5]:
            st.metric(label="Chlorides", value=str(chlorides))
        with stats[6]:
            st.metric(label="Residual sugar", value=str(res_sugar))


        dt = pd.DataFrame([{
            'residual sugar': res_sugar,
            'chlorides': chlorides,
            'total sulfur dioxide': so2,
            'density': density,
            'pH': ph,
            'sulphates': suplhates,
            'alcohol': alcohol
        }])

        # predicting quality
        wine_quality_model = pickle.load(open('wine-quality.pkl', 'rb'))
        prediction = wine_quality_model.predict(dt)
        wine_quality = prediction[0]

        # predicting type
        wine_type_model = pickle.load(open('wine-type.pkl', 'rb'))
        prediction = wine_type_model.predict(dt)
        wine_type = prediction[0]

        if wine_quality == 'low':
            price = '5-9'
            cl_quality = '#037ffc'
        elif wine_quality == 'medium':
            price = '9-19'
            cl_quality = '#cc0000'
        else:
            price = '19-20'
            cl_quality = 'green'

        if wine_type == 'red':
            cl_type = 'red'
        else:
            cl_type = 'yellow'

        st.write("<hr />", unsafe_allow_html=True)

        results = st.columns(3)

        st.write("<hr />", unsafe_allow_html=True)

        with results[0]:
            st.write("Wine type")
            st.write('<span style="color:' + cl_type + ';font-weight:700;font-size:36px">' + wine_type.capitalize() + "</span>", unsafe_allow_html=True)
            st.write("99.9% accuracy")

        with results[1]:
            st.write("Quality")
            st.write('<span style="color:' + cl_quality + ';font-weight:700;font-size:36px">' + wine_quality.capitalize() + "</span>", unsafe_allow_html=True)
            st.write("77% accuracy")

        with results[2]:
            st.write("Price")
            st.write('<span style="font-weight:700;font-size:36px">' + str(price) + "€</style>", unsafe_allow_html=True)




# residual sugar  min: 0.6 max: 65.8
# chlorides  min: 0.009 max: 0.611
# total sulfur dioxide  min: 6.0 max: 440.0
# density  min: 0.98711 max: 1.03898
# pH  min: 2.72 max: 4.01
# sulphates  min: 0.22 max: 2.0
# alcohol  min: 8.0 max: 14.9

with st.sidebar:
    # https://docs.streamlit.io/library/api-reference/widgets/st.slider
    alcohol = st.slider(
        'Alcohol (%)',
        min_value=7, max_value=16, value=9, step=1,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    suplhates = st.slider(
        'Sulphates',
        min_value=0.2, max_value=2.0, value=0.6, step=0.1,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    ph = st.slider(
        'pH',
        min_value=2.5, max_value=4.5, value=3.0, step=0.1,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    density = st.slider(
        'Density',
        min_value=0.9, max_value=1.1, value=0.95, step=0.01,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    so2 = st.slider(
        'Total sulfur dioxide',
        min_value=5, max_value=450, value=30, step=70,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    chlorides = st.slider(
        'Chlorides',
        min_value=0.0, max_value=0.7, value=0.1, step=0.02,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    res_sugar = st.slider(
        'Residual sugar',
        min_value=0.5, max_value=70., value=10.0, step=1.,
        format=None, key=None, help=None, on_change=None, disabled=False, label_visibility="visible"
    )

    st.button(
        'Predict',
        key=None,
        help=None,
        on_click=predict,
        type="primary", disabled=False, use_container_width=True
    )


