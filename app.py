import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO

st.set_page_config(
    page_title="Clasificador de Género",
    page_icon="🔮",
    layout="wide"
)

# ---------------------- Función para convertir imagen a base64 ----------------------
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# ---------------------- Estilos Universidad Externado ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Fondo con gradiente Universidad Externado */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main {
        background-color: transparent;
    }
    
    /* Header personalizado con logos */
    .custom-header {
        background: linear-gradient(135deg, #1a5f3f 0%, #2d9ba4 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: nowrap;
        gap: 1.5rem;
        max-width: 100%;
    }
    
    .logo-container {
        flex: 0 0 auto;
        min-width: 80px;
    }
    
    .logo-box {
        width: 310px;
        height: 210px;
        background: white;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        overflow: hidden;
    }
    
    .logo-box img {
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        padding: 5px;
    }
    
    .logo-icon {
        font-size: 2.5rem;
    }
    
    .title-container {
        flex: 1 1 auto;
        text-align: center;
        min-width: 0;
    }
    
    .header-title {
        margin: 0;
        color: #fcfcfc !important;
        font-size: 4.2rem;
        font-weight: 900;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.9);
        line-height: 1.2;
        letter-spacing: 0.5px;
    }
    
    .header-subtitle {
        margin: 0.5rem 0 0 0;
        color: rgba(255, 255, 255, 0.95);
        font-size: 2.95rem;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Responsive */
    @media (max-width: 600px) {
        .header-content {
            gap: 1rem;
        }
        
        .logo-box {
            width: 80px;
            height: 80px;
        }
        
        .header-title {
            font-size: 2.5rem;
        }
        
        .header-subtitle {
            font-size: 1.85rem;
        }
    }
    
    /* Títulos */
    h1 {
        font-weight: 700;
        text-align: center;
        color: #2d9ba4;
        font-size: 4rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        font-weight: 600;
        color: #2d9ba4;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        text-align: center;
        color: #a8b2d1;
        font-size: 5.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Tarjetas con glassmorphism */
    .card {
        background: rgba(45, 155, 164, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(45, 155, 164, 0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Métricas personalizadas */
    [data-testid="stMetricValue"] {
        font-size: 4rem;
        font-weight: 700;
        color: #FFD700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8b2d1;
        font-weight: 500;
        font-size: 3rem;
    }
    
    [data-testid="metric-container"] {
        background: rgba(45, 155, 164, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(45, 155, 164, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(45, 155, 164, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed rgba(45, 155, 164, 0.5);
    }
    
    [data-testid="stFileUploader"] label {
        color: #a8b2d1 !important;
        font-weight: 500;
        font-size: 3.1rem;
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #2d9ba4 0%, #1a5f3f 100%);
        color: white;
        border-radius: 50px;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 4.5rem;
        box-shadow: 0 4px 15px rgba(45, 155, 164, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(45, 155, 164, 0.6);
        background: linear-gradient(135deg, #1a5f3f 0%, #2d9ba4 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(45, 155, 164, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a8b2d1;
        border-radius: 30px;
        font-weight: 500;
        padding: 0.8rem 1.5rem;
        font-size: 4rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d9ba4 0%, #1a5f3f 100%);
        color: white;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(45, 155, 164, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(45, 155, 164, 0.4);
        color: #a8b2d1;
        font-weight: 500;
    }
    
    /* Info message */
    .stInfo {
        background: rgba(45, 155, 164, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(45, 155, 164, 0.3);
        color: #a8b2d1;
        font-weight: 500;
    }
    
    /* Imágenes */
    img {
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Contenedor de imagen */
    [data-testid="stImage"] {
        background: rgba(45, 155, 164, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 1rem;
        border: 1px solid rgba(45, 155, 164, 0.3);
    }
    
    /* Caption */
    [data-testid="stImageCaption"] {
        color: #a8b2d1;
        font-weight: 500;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2d9ba4 0%, #1a5f3f 100%);
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# ENCABEZADO CON LOGOS
# ============================================


try:
    # 👇 CAMBIA ESTAS RUTAS POR LAS DE TUS LOGOS 👇
    logo_left = Image.open("logo-externado.png")
    logo_right = Image.open("NeuroMInds2.png")
    # 👆 CAMBIA ESTAS RUTAS POR LAS DE TUS LOGOS 👆
    
    logo_left_b64 = image_to_base64(logo_left)
    logo_right_b64 = image_to_base64(logo_right)
    
    st.markdown(f'''
    <div class="custom-header">
        <div class="header-content">
            <div class="logo-container">
                <div class="logo-box">
                    <img src="{logo_left_b64}" alt="Logo Izquierdo">
                </div>
            </div>
            <div class="title-container">
                <h1 class="header-title">Clasificador de Género con IA 🔮 </h1>
                <p class="header-subtitle">Análisis inteligente con visualización de interpretabilidad</p>
            </div>
            <div class="logo-container">
                <div class="logo-box">
                    <img src="{logo_right_b64}" alt="Logo Derecho">
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
except FileNotFoundError as e:
    # Si no encuentra los archivos de logos
    st.error(f"⚠️ No se encontraron los logos. Error: {str(e)}")
    st.info("💡 Verifica que los archivos 'logo-original-white.png' y 'NeuroMInds2.png' estén en la misma carpeta que este archivo .py")
    
    # Mostrar header con emojis como fallback
    st.markdown('''
    <div class="custom-header">
        <div class="header-content">
            <div class="logo-container">
                <div class="logo-box">
                    <span class="logo-icon">🎓</span>
                </div>
            </div>
            <div class="title-container">
                <h1 class="header-title">Clasificador de Género con IA</h1>
                <p class="header-subtitle">Análisis inteligente con visualización de interpretabilidad</p>
            </div>
            <div class="logo-container">
                <div class="logo-box">
                    <span class="logo-icon">🤖</span>
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

except Exception as e:
    # Cualquier otro error
    st.error(f"⚠️ Error al cargar los logos: {str(e)}")
    
    # Mostrar header con emojis como fallback
    st.markdown('''
    <div class="custom-header">
        <div class="header-content">
            <div class="logo-container">
                <div class="logo-box">
                    <span class="logo-icon">🎓</span>
                </div>
            </div>
            <div class="title-container">
                <h1 class="header-title">Clasificador de Género con IA</h1>
                <p class="header-subtitle">Análisis inteligente con visualización de interpretabilidad</p>
            </div>
            <div class="logo-container">
                <div class="logo-box">
                    <span class="logo-icon">🤖</span>
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

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
# Espaciado
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### 📤 Sube una imagen para analizar")  # Título más grande
#           ^^^
#        Cambia ### a ## (más grande) o #### (más pequeño)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

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
        st.image(img, width=500)

    with col2:
        st.markdown("## 🎯 Resultados del Análisis")
        
        # Métricas en subcolomnas
        met1, met2 = st.columns(2)
        with met1:
            st.metric("### 👩 Mujer", f"{prob_female*100:.1f}%")
        with met2:
            st.metric("### 👨 Hombre", f"{prob_male*100:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Resultado final
        emoji = '👨' if pred_class == 'Hombre' else '👩'
        st.success(f"### {emoji} Clasificación: **{pred_class}**")
        
        # Barra de confianza
        confidence = max(prob_male, prob_female) * 100
        st.markdown(f'<p style="font-size: 2.5rem; font-weight: 600; color: #ffffff;">Confianza: {confidence:.1f}%</p>', unsafe_allow_html=True)
#                      
        st.progress(confidence / 100)

    # ---------------------- Espaciado ----------------------
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------- Pestañas Interpretabilidad ----------------------
    st.markdown("## 🔍 Análisis de Interpretabilidad")
    st.markdown('<p class="subtitle">Descubre qué regiones de la imagen influyeron en la decisión del modelo</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔥 Grad-CAM (Mapas de Calor)", "🌈 Saliency Map (Sensibilidad)"])

    with tab1:
        st.markdown("## Regiones de Mayor Influencia")
        st.markdown("### Los mapas de calor muestran las áreas que más contribuyeron a la clasificación")
        
        target_layers_idx = [1, 3, 4]
        cols = st.columns(len(target_layers_idx))
        
        for i, idx_layer in enumerate(target_layers_idx):
            heatmap = grad_cam_sequential(model, img_input, class_index=class_index, target_layer_index=idx_layer)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            superimposed_img = cv2.addWeighted(img, 0.6, cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET), 0.4, 0)
            
            with cols[i]:
                st.image(superimposed_img, caption=f"Capa: {model.layers[idx_layer].name}", width=400)

    with tab2:
        st.markdown("## Análisis de Sensibilidad por Píxel")
        st.markdown("### Visualización de qué píxeles tienen mayor impacto en la predicción")
        
        sal_map = saliency_map(model, img_input, class_index=class_index)
        sal_map_resized = cv2.resize(sal_map, (img.shape[1], img.shape[0]))
        sal_map_img = np.uint8(255 * sal_map_resized)
        sal_map_img = cv2.applyColorMap(sal_map_img, cv2.COLORMAP_JET)
        superimposed_sal = cv2.addWeighted(img, 0.6, sal_map_img, 0.4, 0)
        
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            st.image(superimposed_sal, width=450)

else:
    # Mensaje cuando no hay imagen
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👆 Por favor, sube una imagen para comenzar el análisis")
