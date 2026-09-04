# ============================================
# API DE CLASIFICACIÓN DE CABELLO
# Versión definitiva - Compatible con Keras 2.15.0
# ============================================

import os
import json
import io
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

app = Flask(__name__)

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Usar el archivo .h5 (compatible con Keras 2)
MODEL_PATH = os.path.join(BASE_DIR, "hair_classifier_v2_compatible.h5")
METADATA_PATH = os.path.join(BASE_DIR, "hair_classifier_v2_metadata.json")

# --- CARGA DEL MODELO ---
print("🔄 Cargando modelo .h5...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    raise

# --- CARGA DE METADATOS ---
try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    class_names = metadata['class_names']
    print(f"📊 Clases: {class_names}")
except Exception as e:
    print(f"❌ Error al cargar metadatos: {e}")
    raise

# --- FUNCIÓN DE PREPROCESAMIENTO ---
def preprocess_image(image_bytes):
    """
    Convierte cualquier imagen a RGB y la preprocesa para el modelo.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Error al abrir la imagen: {e}")
    
    # Convertir a RGB (maneja RGBA, P, etc.)
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar y preprocesar
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch

# --- ENDPOINTS ---

@app.route('/', methods=['GET'])
def home():
    """Endpoint raíz - Información de la API"""
    return jsonify({
        'message': 'API de Clasificación de Cabello',
        'version': '2.0',
        'endpoints': {
            '/predict': 'POST - Envía una imagen para clasificar',
            '/health': 'GET - Verifica el estado del servicio'
        },
        'classes': class_names
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud para verificar que el servicio está activo"""
    return jsonify({
        'status': 'healthy',
        'model': 'hair_classifier_v2_compatible.h5',
        'classes': class_names,
        'tensorflow_version': '2.15.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de predicción.
    Espera un archivo de imagen en el campo 'image'.
    """
    try:
        # Verificar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se proporcionó ninguna imagen. Envía un archivo en el campo "image".'
            }), 400
        
        file = request.files['image']
        
        # Verificar que el archivo no esté vacío
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'El nombre del archivo está vacío.'
            }), 400
        
        # Leer y preprocesar la imagen
        image_bytes = file.read()
        img_batch = preprocess_image(image_bytes)
        
        # Hacer la predicción
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Obtener Top-3 predicciones
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'class': class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        # Respuesta exitosa
        return jsonify({
            'success': True,
            'prediction': class_names[predicted_class],
            'confidence': confidence,
            'top_predictions': top_predictions
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Error en el procesamiento de la imagen: {str(e)}'
        }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error interno: {str(e)}'
        }), 500

# --- INICIO DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)