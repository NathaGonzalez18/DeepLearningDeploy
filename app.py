import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Clasificador de Género",
    page_icon="🟣",
    layout="wide"
)

# ---------------------- Estilos ----------------------
st.markdown("""
<style>
h1, h2, h3, h4 {
    font-weight: 600;
    text-align: center;
    color: #C084FC;
}
[class^="stMetric"] {
    background-color: #1a1b1e !important;
    border-radius: 10px;
    padding: 15px;
}
.uploadedFile {
    background-color: #1a1b1e !important;
    padding: 12px;
    border-radius: 8px;
}
img {
    border-radius: 12px;
}
.stButton>button {
    background: linear-gradient(90deg, #8b5cf6, #c084fc);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
.stButton>button:hover {
    opacity: 0.88;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Cargar modelo ----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modelo_genero.h5")
    return model

model = load_model()
IMG_SIZE = 224

# ---------------------- Funciones ----------------------
def grad_cam_sequential(model, image_tensor, class_index, target_layer_index):
    with tf.GradientTape() as tape:
        x = image_tensor
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == target_layer_index:
                conv_outputs = x
        predictions = x
        loss = predictions[:, 0] if class_index == 1 else 1 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def saliency_map(model, image_tensor, class_index):
    image_tensor = tf.convert_to_tensor(image_tensor)
    image_tensor = tf.Variable(image_tensor)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        pred = model(image_tensor)
        loss = pred[:, 0] if class_index == 1 else 1 - pred[:, 0]
    grads = tape.gradient(loss, image_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
    return saliency.numpy()

# ---------------------- Layout principal ----------------------
st.title("Clasificación de Género con Interpretabilidad (Grad-CAM & Saliency Map)")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocesar para modelo
    img_input = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Predicción
    pred = model.predict(img_input)
    prob_male = float(pred[0, 0])
    prob_female = 1 - prob_male
    pred_class = "Hombre" if prob_male > 0.5 else "Mujer"
    class_index = 1 if pred_class == "Hombre" else 0

    # ---------------------- Columnas ----------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Imagen cargada", width=400)

    with col2:
        st.subheader("Resultado de la Clasificación")
        st.metric("Probabilidad Mujer", f"{prob_female:.3f}")
        st.metric("Probabilidad Hombre", f"{prob_male:.3f}")
        st.success(f"Clasificación: **{pred_class}** {'👨' if pred_class=='Hombre' else '👩'}")

    # ---------------------- Pestañas Grad-CAM y Saliency ----------------------
    tab1, tab2 = st.tabs(["🔥 Grad-CAM", "🌈 Saliency Map"])

    with tab1:
        st.subheader("Grad-CAM: Regiones que más influyeron en la decisión")
        target_layers_idx = [0, 3, 4]  # Ajusta según tu modelo
        # Crear columnas dinámicamente según la cantidad de Grad-CAMs
        cols = st.columns(len(target_layers_idx))
        for i, idx_layer in enumerate(target_layers_idx):
            heatmap = grad_cam_sequential(model, img_input, class_index=class_index, target_layer_index=idx_layer)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            superimposed_img = cv2.addWeighted(img, 0.6, cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET), 0.4, 0)
            cols[i].image(superimposed_img, caption=f"{model.layers[idx_layer].name}", width=250)


    with tab2:
        st.subheader("Saliency Map: Sensibilidad por píxel")
        sal_map = saliency_map(model, img_input, class_index=class_index)
        sal_map_resized = cv2.resize(sal_map, (img.shape[1], img.shape[0]))
        sal_map_img = np.uint8(255 * sal_map_resized)
        sal_map_img = cv2.applyColorMap(sal_map_img, cv2.COLORMAP_JET)
        superimposed_sal = cv2.addWeighted(img, 0.6, sal_map_img, 0.4, 0)
        st.image(superimposed_sal, caption="Saliency Map", width=300)



