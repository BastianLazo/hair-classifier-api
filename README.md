# API de Clasificación de Cabello

## Descripción
API para clasificar imágenes de cabello en 5 categorías:
- Straight
- Wavy
- Curly
- Dreadlocks
- Kinky

## Endpoints

### `GET /`
Información de la API.

### `GET /health`
Verifica el estado del servicio.

### `POST /predict`
Clasifica una imagen de cabello.

**Parámetros:**
- `image`: Archivo de imagen (JPG, PNG) en el campo `image`.

**Ejemplo con `curl`:**
```bash
curl -X POST -F "image=@imagen.jpg" https://tu-servicio.onrender.com/predict