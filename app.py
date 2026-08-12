import os
import numpy as np
import scipy.io
import streamlit as st
import time

from scipy.signal import butter, filtfilt, welch
from scipy.integrate import simpson
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title(" Real-Time Brain State Dashboard ")


@st.cache_resource
def load_and_train():

    folder = r"C:\Users\user\anaconda3\envs\eeg_env\EEG Data\EEG Data"
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]

    all_epochs = []
    all_labels = []

    fs = 128

    for file in files:
        data = scipy.io.loadmat(os.path.join(folder, file))
        eeg = data["o"][0][0]["data"]

        # CHANNEL SELECT
        eeg = eeg[:, 2:16]
        eeg = np.delete(eeg, [12, 13], axis=1)

        # REFERENCING
        eeg = eeg - np.mean(eeg, axis=1, keepdims=True)

        # FILTER
        b, a = butter(4, [0.5/(fs/2), 40/(fs/2)], btype='band')
        eeg = filtfilt(b, a, eeg, axis=0)

        # ICA
        ica = FastICA(n_components=12, random_state=0)
        comp = ica.fit_transform(eeg)
        comp[:, [2,6,10]] = 0
        eeg = ica.inverse_transform(comp)

        # EPOCHING
        epoch_len = 2 * fs
        n = eeg.shape[0] // epoch_len
        epochs = eeg[:n*epoch_len].reshape(n, epoch_len, 12)

        # LABELING (FIXED)
        for ep in epochs:
            f, psd = welch(ep[:,0], fs=fs)

            delta = simpson(psd[(f>=0.5)&(f<=4)], f[(f>=0.5)&(f<=4)])
            theta = simpson(psd[(f>=4)&(f<=8)], f[(f>=4)&(f<=8)])
            alpha = simpson(psd[(f>=8)&(f<=13)], f[(f>=8)&(f<=13)])
            beta  = simpson(psd[(f>=13)&(f<=30)], f[(f>=13)&(f<=30)])

            total = delta + theta + alpha + beta + 1e-6

            d, t, a, b = delta/total, theta/total, alpha/total, beta/total

            # RELATIVE DOMINANCE (NO HARD THRESHOLD)
            if b > a and b > t:
                label = "Focused"
            elif a > b and a > t:
                label = "Relaxed"
            elif t > b:
                label = "Drowsy"
            else:
                label = "Neutral"

            all_epochs.append(ep)
            all_labels.append(label)

    X = np.array(all_epochs)
    labels = np.array(all_labels)

    # OPTIONAL: REMOVE NEUTRAL
    valid_idx = labels != "Neutral"
    X = X[valid_idx]
    labels = labels[valid_idx]

    # ENCODE
    le = LabelEncoder()
    y = le.fit_transform(labels)
    y_cat = to_categorical(y)

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, stratify=y, random_state=42
    )

    # CLASS WEIGHTS
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(enumerate(weights))

    # MODEL
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(256,12)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        epochs=6,
        batch_size=32,
        class_weight=class_weights,
        verbose=0
    )

    return X, model, le, fs

epochs, model, le, fs = load_and_train()


# DASHBOARD UI

col1, col2, col3 = st.columns(3)
state_box = col1.empty()
conf_box = col2.empty()
alert_box = col3.empty()

mid1, mid2 = st.columns(2)
bar_box = mid1.empty()
pie_box = mid2.empty()

explain_box = st.empty()


# REAL-TIME SIMULATION

if st.button("▶ Start Monitoring"):

    for i in range(len(epochs)):

        ep = epochs[i]

        pred = model.predict(ep.reshape(1,256,12))[0]

        # TOP 2 STATES
        top2 = np.argsort(pred)[-2:][::-1]

        state = le.classes_[top2[0]]
        confidence = pred[top2[0]]

        # EEG BAND POWER
        f, psd = welch(ep[:,0], fs=fs)

        delta = simpson(psd[(f>=0.5)&(f<=4)], f[(f>=0.5)&(f<=4)])
        theta = simpson(psd[(f>=4)&(f<=8)], f[(f>=4)&(f<=8)])
        alpha = simpson(psd[(f>=8)&(f<=13)], f[(f>=8)&(f<=13)])
        beta  = simpson(psd[(f>=13)&(f<=30)], f[(f>=13)&(f<=30)])

        total = delta+theta+alpha+beta+1e-6
        vals = [delta/total, theta/total, alpha/total, beta/total]

        # UI
        state_box.markdown(f"## State: **{state}**")
        conf_box.metric("Confidence", f"{confidence*100:.1f}%")

        alert_box.info(f"Alt: {le.classes_[top2[1]]} ({pred[top2[1]]*100:.1f}%)")

        # BAR
        bar = go.Figure([go.Bar(
            x=["Delta","Theta","Alpha","Beta"],
            y=vals,
            text=[f"{v:.2f}" for v in vals],
            textposition='auto'
        )])
        bar.update_layout(title="Brainwave Strength")
        bar_box.plotly_chart(bar, key=f"bar{i}")

        # PIE
        pie = go.Figure(data=[go.Pie(
            labels=["Delta","Theta","Alpha","Beta"],
            values=vals,
            hole=0.4
        )])
        pie.update_layout(title="Brainwave Distribution")
        pie_box.plotly_chart(pie, key=f"pie{i}")

        # EXPLANATION
        explanation = f"""
###  Why this state?

- Delta: {vals[0]:.2f}
- Theta: {vals[1]:.2f}
- Alpha: {vals[2]:.2f}
- Beta: {vals[3]:.2f}
"""

        if state == "Focused":
            explanation += " Beta dominant → Active thinking"
        elif state == "Relaxed":
            explanation += " Alpha dominant → Calm state"
        elif state == "Drowsy":
            explanation += " Theta dominant → Sleepy"
        else:
            explanation += " Mixed → Neutral"

        explain_box.markdown(explanation)

        time.sleep(0.5)
