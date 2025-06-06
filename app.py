import streamlit as st
import pdfplumber
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- Configuración de Rutas (¡AJUSTADO PARA HUGGING FACE HUB!) ---
# Nombre del repositorio del modelo en Hugging Face Hub
# ¡Asegúrate de que 'Yordano35/clasificador-licitaciones-juridico' sea EXACTAMENTE el nombre que usaste al subirlo!
HF_MODEL_NAME_OR_PATH = "Yordano35/clasificador-licitaciones-juridico"

# Rutas para las imágenes de métricas y logos (estas sí son locales en tu repositorio de GitHub)
LOG_LOSS_PLOT = "./assets/images/loss_plot.png"
LOG_EVAL_METRICS_PLOT = "./assets/images/eval_metrics_plot.png"
LOGO_IMAGE = "./assets/images/logo_licitaciones.png"
DECORATIVE_IMAGE = "./assets/images/decorative_image.png"

# --- Configuración de Gemini (¡Importante para despliegue!) ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.warning("¡Advertencia! La API Key de Gemini no se encontró en st.secrets. "
                "Usando una variable de entorno o placeholder para desarrollo local. "
                "Por favor, configura st.secrets['GEMINI_API_KEY'] para despliegue.")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_ENV", "TU_API_KEY_AQUI_PARA_PRUEBAS_LOCALES")


genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Cargar Modelo Gemini ---
@st.cache_resource
def load_gemini_model():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "TU_API_KEY_AQUI_PARA_PRUEBAS_LOCALES":
        st.error("API Key de Gemini no configurada. Por favor, revisa tus 'Secrets' o la variable de entorno.")
        return None
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        st.success("🤖 Modelo Gemini cargado.")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo Gemini: {e}. Asegúrate de que tu API Key sea válida.")
        return None

gemini_model = load_gemini_model()

# --- Cargar el Modelo Clasificador de Hugging Face ---
@st.cache_resource(show_spinner="Cargando modelo clasificador de licitaciones...")
def load_classifier_model():
    try:
        # Aquí es donde cambiamos la carga: de ruta local a Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME_OR_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME_OR_PATH)
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            device=0 if torch.cuda.is_available() else -1 # Usa GPU si está disponible
        )
        st.success("🚀 Modelo clasificador cargado desde Hugging Face Hub.")
        return classifier
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo clasificador: {e}. "
                 f"Asegúrate de que el nombre del repositorio en Hugging Face Hub ({HF_MODEL_NAME_OR_PATH}) "
                 f"sea correcto y que el modelo esté completo y sea accesible.")
        st.info("Verifica el nombre del repositorio en https://huggingface.co/models")
        return None

classifier = load_classifier_model()

# --- Funciones de Preprocesamiento y Resumen (con @st.cache_data) ---

@st.cache_data(show_spinner="Extrayendo texto del PDF...")
def extract_text_from_pdf(pdf_file):
    """Extrae texto de un archivo PDF subido por Streamlit."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error al extraer texto del PDF: {e}")
        return None

@st.cache_data(show_spinner="Preparando resumen del documento (esto puede tardar unos segundos)...")
def summarize_text_with_gemini(text_content, model):
    """Resume un texto usando el modelo Gemini."""
    if not model:
        return " Modelo Gemini no disponible debido a un error de configuración."

    custom_prompt = (
        "Eres un asistente experto en resumir documentos PDF de procesos licitatorios. "
        "Tu objetivo es generar resúmenes concisos, estructurados y altamente informativos, "
        "enfocándote específicamente en las exigencias y aspectos relevantes de las áreas: "
        "jurídica, financiera, técnica y experiencia.\n\n"
        "Considera las siguientes secciones clave del documento:\n"
        "- Todas las exigencias jurídicas\n"
        "- Todas las exigencias financieras\n"
        "- Todas las exigencias técnicas\n"
        "- Exigencias de experiencia\n\n"
        "Por favor, genera un resumen global y detallado del siguiente texto en español, "
        "organizando la información bajo los puntos clave mencionados. Si no encuentras información relevante para una sección, simplemente no la menciones o indica que no aplica.:\n\n"
    )

    MAX_CHARS_FOR_GEMINI = 350000

    if len(text_content) > MAX_CHARS_FOR_GEMINI:
        st.warning(f"El documento es muy largo ({len(text_content)} caracteres). Se procesarán los primeros {MAX_CHARS_FOR_GEMINI} caracteres para el resumen. Esto podría afectar la completitud del resumen.")
        text_content_for_gemini = text_content[:MAX_CHARS_FOR_GEMINI]
    else:
        text_content_for_gemini = text_content

    try:
        response = model.generate_content(custom_prompt + text_content_for_gemini)
        return response.text
    except Exception as e:
        st.error(f"❌ Error al generar resumen con Gemini: {e}")
        st.info("Considera que el PDF puede ser demasiado largo o tu API Key de Gemini podría tener problemas.")
        return f"Error en la API de Gemini: {e}"

# --- Mapeo de Etiquetas ---
label_map_prediction = {0: "RECHAZADA", 1: "APROBADA"}

# --- Configuración de la Interfaz de Streamlit ---
st.set_page_config(
    page_title="Clasificador de Licitaciones",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título y Decoración Principal ---
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    try:
        logo = Image.open(LOGO_IMAGE)
        st.image(logo, width=120)
    except FileNotFoundError:
        st.markdown("## 📄")

with col2:
    st.title("Clasificador Inteligente de Licitaciones")
    st.markdown("""
        **Analiza y predice el estado de tus documentos de licitación al instante.**
        Esta aplicación utiliza inteligencia artificial para clasificar automáticamente
        si una licitación es `APROBADA` o `RECHAZADA`, optimizando la toma de decisiones.
    """)

with col3:
    try:
        decorative_img = Image.open(DECORATIVE_IMAGE)
        st.image(decorative_img, width=120)
    except FileNotFoundError:
        st.markdown("## 📊")

st.markdown("---") # Divisor visual

# --- Sección Principal de la Aplicación ---
st.header("Sube un Nuevo Documento para Analizar")
uploaded_file = st.file_uploader("Arrastra y suelta tu archivo PDF aquí o haz clic para buscar:", type="pdf", help="El PDF será procesado y clasificado.")

if uploaded_file is not None:
    st.info(f"📄 Archivo '{uploaded_file.name}' subido. Iniciando análisis...")

    # Usamos un contenedor para el feedback del proceso
    with st.container():
        # Extraer texto
        pdf_text = extract_text_from_pdf(uploaded_file)
        if not pdf_text:
            st.error("No se pudo extraer texto del PDF. Por favor, intenta con otro archivo o verifica su formato.")
            st.stop()

        # Generar resumen con Gemini (en segundo plano, no se muestra)
        summary = summarize_text_with_gemini(pdf_text, gemini_model)

        # Clasificar la licitación
        if classifier:
            st.subheader("📊 Resultado de la Clasificación")
            # Asegurarse de que el resumen no esté vacío antes de clasificar
            if summary and len(summary.strip()) > 0 and not summary.startswith("Error"):

                with st.spinner('Procesando y prediciendo el estado de la licitación...'):
                    prediction = classifier(summary)

                predicted_label_raw = prediction[0]['label']
                predicted_score = prediction[0]['score']

                try:
                    # Intenta convertir la etiqueta si tiene formato 'LABEL_X'
                    numeric_label = int(predicted_label_raw.split('_')[1])
                except (IndexError, ValueError):
                    # Si no tiene ese formato, o si el split falla, usa las etiquetas directas
                    if predicted_label_raw == "APROBADA":
                        numeric_label = 1
                    elif predicted_label_raw == "RECHAZADA":
                        numeric_label = 0
                    else:
                        numeric_label = -1 # Valor para desconocido

                final_prediction_text = label_map_prediction.get(numeric_label, "Desconocido")

                col_pred_result, col_pred_prob = st.columns([1, 1])
                with col_pred_result:
                    if final_prediction_text == "APROBADA":
                        st.success(f"### Decisión: **{final_prediction_text}** ")
                    else:
                        st.error(f"### Decisión: **{final_prediction_text}** ")
                with col_pred_prob:
                    st.metric(label="Confianza de la Predicción", value=f"{predicted_score:.2%}")

                st.info("La clasificación se realiza sobre un resumen generado por IA para mantener la consistencia del modelo.")
            else:
                st.warning("No se pudo clasificar el documento porque el resumen interno no es válido. Revisa el PDF o la configuración de Gemini.")
        else:
            st.error("El modelo clasificador no se pudo cargar. No se puede realizar la predicción.")

st.markdown("---") # Divisor visual

# --- Sección de Métricas de Entrenamiento ---
st.header("📈 Métricas Clave del Proceso de Entrenamiento")
st.markdown("Comprende el rendimiento y el aprendizaje de nuestro modelo de clasificación:")

# Usar columnas para organizar los gráficos
metrics_col_left, metrics_col_right = st.columns(2)

with metrics_col_left:
    st.subheader("Pérdida (Loss) del Modelo")
    if os.path.exists(LOG_LOSS_PLOT):
        st.image(LOG_LOSS_PLOT, caption="Pérdida de Entrenamiento y Validación por Paso", use_column_width=True)
    else:
        st.warning(f"No se encontró el gráfico de pérdida en {LOG_LOSS_PLOT}.")
        st.info("Asegúrate de haber copiado `loss_plot.png` a la carpeta `assets/images/` en tu repositorio.")

with metrics_col_right:
    st.subheader("Métricas de Evaluación del Modelo")
    if os.path.exists(LOG_EVAL_METRICS_PLOT):
        st.image(LOG_EVAL_METRICS_PLOT, caption="Métricas de Evaluación por Época (Accuracy, F1-Score, Precision, Recall)", use_column_width=True)
    else:
        st.warning(f"No se encontró el gráfico de métricas de evaluación en {LOG_EVAL_METRICS_PLOT}.")
        st.info("Asegúrate de haber copiado `eval_metrics_plot.png` a la carpeta `assets/images/` en tu repositorio.")

st.markdown("---")
st.markdown("""
    Desarrollado por [grupo Tech IA Anderson y compañeros] | [2025]
    """)

# Puedes añadir una imagen al final para un toque final
try:
    footer_image = Image.open(DECORATIVE_IMAGE) # O una imagen diferente
    st.image(footer_image, width=80)
except FileNotFoundError:
    pass # No mostrar nada si la imagen no existe