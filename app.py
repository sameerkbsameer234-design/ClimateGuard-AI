
import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go


st.set_page_config(
    page_title="ClimateGuard AI",
    layout="wide"
)


st.title("🌍 ClimateGuard AI")

st.subheader(
    "Multi-Feature Climate Forecasting System"
)


df = pd.read_csv(
    "GlobalWeatherRepository.csv"
)


features = [

    'temperature_celsius',

    'humidity',

    'pressure_mb',

    'wind_kph',

    'precip_mm',

    'visibility_km',

    'uv_index',

    'cloud',

    'feels_like_celsius',

    'gust_kph'
]

df = df[features]

df.dropna(inplace=True)


scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)


model = load_model(
    "Climateguard_ai_model.keras"
)


SEQ_LENGTH = 10

X = []

for i in range(
    SEQ_LENGTH,
    len(scaled_data)
):

    X.append(
        scaled_data[
            i-SEQ_LENGTH:i
        ]
    )

X = np.array(X)


future_days = st.slider(

    "Select Forecast Days",

    1,

    30,

    7
)


last_sequence = X[-1]

future_predictions = []

current_sequence = last_sequence.reshape(

    1,

    SEQ_LENGTH,

    len(features)
)

for _ in range(future_days):

    next_pred = model.predict(

        current_sequence,

        verbose=0
    )

    future_predictions.append(
        next_pred[0][0]
    )

    next_step = current_sequence[
        0,-1,:
    ].copy()

    next_step[0] = next_pred[0][0]

    current_sequence = np.append(

        current_sequence[:,1:,:],

        [[next_step]],

        axis=1
    )

future_predictions = np.array(
    future_predictions
).reshape(-1,1)


dummy_future = np.zeros(

    (
        len(future_predictions),

        len(features)
    )
)

dummy_future[:,0] = (
    future_predictions.flatten()
)

future_actual = scaler.inverse_transform(

    dummy_future

)[:,0]


risk_threshold = np.mean(
    future_actual
)

risk_labels = np.where(

    future_actual > risk_threshold,

    "High Risk",

    "Moderate Risk"
)

forecast_df = pd.DataFrame({

    'Day': np.arange(

        1,

        future_days + 1
    ),

    'PredictedTemperature':
    future_actual,

    'RiskLevel':
    risk_labels
})


st.subheader(
    "📊 Forecast Data"
)

st.dataframe(forecast_df)


fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=forecast_df['Day'],

        y=forecast_df[
            'PredictedTemperature'
        ],

        mode='lines+markers',

        name='Forecast'
    )
)

fig.update_layout(

    title="Future Climate Forecast",

    xaxis_title="Days",

    yaxis_title="Temperature"
)

st.plotly_chart(

    fig,

    use_container_width=True
)


high_risk = len(

    forecast_df[
        forecast_df['RiskLevel']
        == "High Risk"
    ]
)

moderate_risk = len(

    forecast_df[
        forecast_df['RiskLevel']
        == "Moderate Risk"
    ]
)

col1, col2 = st.columns(2)

col1.metric(
    "🔥 High Risk Days",
    high_risk
)

col2.metric(
    "⚠ Moderate Risk Days",
    moderate_risk
)


st.success(
    "✅ Climate Forecast Generated Successfully!"
)

st.markdown("---")

st.markdown(
    "Developed using LSTM Deep Learning"
)
