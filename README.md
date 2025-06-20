# PDI_TP3_Frattini_Peralta_Prado
Procesamiento de video con OpenCV
# Detección de Carriles en Video con OpenCV

**Autores:** Matías Prado, Gianfranco Frattini, Alejandro Peralta  
**Trabajo Práctico – Procesamiento de Imágenes y Visión por Computadora**

Este proyecto implementa un sistema de detección de carriles en videos utilizando técnicas de procesamiento digital de imágenes con Python y OpenCV. El sistema identifica bordes mediante el algoritmo de Canny, aplica una región de interés (ROI), y detecta líneas con la Transformada de Hough Probabilística. Posteriormente, clasifica y promedia líneas correspondientes al carril izquierdo y derecho, superponiéndolas sobre el video original con colores intensos para facilitar su visualización.

## Tecnologías utilizadas

- Python 3
- OpenCV
- NumPy

## Ejecución

1. Instalar dependencias:
```
pip install opencv-python numpy
```

2. Ejecutar el script principal:
```
python detectar_carriles.py
```

Los resultados se guardan en la carpeta `procesos/`, incluyendo imágenes del primer cuadro procesado y un video con las líneas de carril detectadas.

Este trabajo busca aplicar conceptos teóricos a un problema real relacionado con la visión por computadora y la conducción asistida.
