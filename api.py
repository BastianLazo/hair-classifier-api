# ============================================
# API PARA RENDER - CON NUEVO MODELO KERAS 2
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ MODELO NUEVO (Keras 2)
MODEL_PATH = os.path.join(BASE_DIR, "hair_classifier_v2_keras2.h5")
METADATA_PATH = os.path.join(BASE_DIR, "hair_classifier_v2_metadata.json")

print("🔄 Cargando modelo (Keras 2)...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    raise

# Cargar metadatos
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
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Error al abrir la imagen: {e}")
    
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch

# --- ENDPOINTS ---
@app.route('/', methods=['GET'])
def home():
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
    return jsonify({
        'status': 'healthy',
        'model': 'hair_classifier_v2_keras2.h5',
        'classes': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se proporcionó ninguna imagen.'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'El nombre del archivo está vacío.'
            }), 400
        
        image_bytes = file.read()
        img_batch = preprocess_image(image_bytes)
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'class': class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
