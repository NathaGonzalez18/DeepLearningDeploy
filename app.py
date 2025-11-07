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

# ---------------------- Estilos Mejorados ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main {
        background-color: transparent;
    }
    
    h1 {
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    h2, h3 {
        font-weight: 600;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #e9d5ff;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Tarjetas con glassmorphism */
    .card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1.5rem;
    }
    
    /* Métricas personalizadas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e9d5ff;
        font-weight: 500;
        font-size: 1rem;
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 50px;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #e9d5ff;
        border-radius: 10px;
        font-weight: 500;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(74, 222, 128, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(74, 222, 128, 0.3);
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Imágenes */
    img {
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* Contenedor de imagen */
    [data-testid="stImage"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Caption */
    [data-testid="stImageCaption"] {
        color: #e9d5ff;
        font-weight: 500;
        text-align: center;
        margin-top: 0.5rem;
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
        loss = predictions[:,0] if class_index == 1 else 1 - predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
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
st.title("🔮 Clasificador de Género con IA")
st.markdown('<p class="subtitle">Análisis inteligente con visualización de interpretabilidad</p>', unsafe_allow_html=True)

# Espaciado
st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Sube una imagen para analizar", type=["jpg", "jpeg", "png"])

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

    # ---------------------- Espaciado ----------------------
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------- Columnas Principales ----------------------
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📸 Imagen Original")
        st.image(img, width=400)

    with col2:
        st.markdown("### 🎯 Resultados del Análisis")
        
        # Métricas en subcolomnas
        met1, met2 = st.columns(2)
        with met1:
            st.metric("👩 Mujer", f"{prob_female*100:.1f}%")
        with met2:
            st.metric("👨 Hombre", f"{prob_male*100:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Resultado final
        emoji = '👨' if pred_class == 'Hombre' else '👩'
        st.success(f"### {emoji} Clasificación: **{pred_class}**")
        
        # Barra de confianza
        confidence = max(prob_male, prob_female) * 100
        st.markdown(f"**Confianza:** {confidence:.1f}%")
        st.progress(confidence / 100)

    # ---------------------- Espaciado ----------------------
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------- Pestañas Interpretabilidad ----------------------
    st.markdown("## 🔍 Análisis de Interpretabilidad")
    st.markdown('<p class="subtitle">Descubre qué regiones de la imagen influyeron en la decisión del modelo</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔥 Grad-CAM (Mapas de Calor)", "🌈 Saliency Map (Sensibilidad)"])

    with tab1:
        st.markdown("### Regiones de Mayor Influencia")
        st.markdown("Los mapas de calor muestran las áreas que más contribuyeron a la clasificación")
        
        target_layers_idx = [1, 3, 4]
        cols = st.columns(len(target_layers_idx))
        
        for i, idx_layer in enumerate(target_layers_idx):
            heatmap = grad_cam_sequential(model, img_input, class_index=class_index, target_layer_index=idx_layer)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            superimposed_img = cv2.addWeighted(img, 0.6, cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET), 0.4, 0)
            
            with cols[i]:
                st.image(superimposed_img, caption=f"Capa: {model.layers[idx_layer].name}", width=350)

    with tab2:
        st.markdown("### Análisis de Sensibilidad por Píxel")
        st.markdown("Visualización de qué píxeles tienen mayor impacto en la predicción")
        
        sal_map = saliency_map(model, img_input, class_index=class_index)
        sal_map_resized = cv2.resize(sal_map, (img.shape[1], img.shape[0]))
        sal_map_img = np.uint8(255 * sal_map_resized)
        sal_map_img = cv2.applyColorMap(sal_map_img, cv2.COLORMAP_JET)
        superimposed_sal = cv2.addWeighted(img, 0.6, sal_map_img, 0.4, 0)
        
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            st.image(superimposed_sal, use_container_width=True)

else:
    # Mensaje cuando no hay imagen
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👆 Por favor, sube una imagen para comenzar el análisis")