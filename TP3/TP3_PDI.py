import cv2
import numpy as np
import os

def procesar_video(ruta_video, salida_video):
    # Carga el video de entrada
    cap = cv2.VideoCapture(ruta_video)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepara el objeto para guardar el video procesado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(salida_video, fourcc, fps, (ancho, alto))

    # Crea carpeta para guardar imágenes del primer frame
    nombre_base = os.path.splitext(os.path.basename(ruta_video))[0]
    carpeta_salida = os.path.join("procesos", nombre_base)
    os.makedirs(carpeta_salida, exist_ok=True)

    primer_frame = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesa el frame para detectar carriles
        resultado = detectar_carril(frame, guardar_imagenes=primer_frame, carpeta_salida=carpeta_salida)
        out.write(resultado)
        primer_frame = False

        # Muestra el resultado en tiempo real
        cv2.imshow('Detección de Carril', resultado)
        if cv2.waitKey(1) == ord('q'):
            break

    # Libera recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detectar_carril(frame, guardar_imagenes=False, carpeta_salida=None):
    # Guarda el frame original si es necesario
    if guardar_imagenes:
        cv2.imwrite(os.path.join(carpeta_salida, "original.jpg"), frame)

    # Conversión a escala de grises y reducción de ruido
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    canny = cv2.Canny(blur, 70, 150)

    # Define la región de interés en forma de trapecio
    altura, ancho = canny.shape
    vertices = np.array([[ 
        (int(ancho * 0.1), int(altura * 0.95)),
        (int(ancho * 0.45), int(altura * 0.6)),
        (int(ancho * 0.55), int(altura * 0.6)),
        (int(ancho * 0.9), int(altura * 0.95))
    ]], dtype=np.int32)

    # Crea máscara para filtrar la región de interés
    mascara = np.zeros_like(canny)
    cv2.fillPoly(mascara, vertices, 255)
    roi = cv2.bitwise_and(canny, mascara)

    # Detecta líneas con Hough Transform
    lineas = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=60, maxLineGap=120)
    imagen_lineas = np.zeros_like(frame)

    izquierda = []
    derecha = []

    # Clasifica líneas según su pendiente
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            pendiente = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(pendiente) > 0.5:
                if pendiente < 0 and x1 < ancho / 2 and x2 < ancho / 2:
                    izquierda.append((x1, y1, x2, y2))
                elif pendiente > 0 and x1 > ancho / 2 and x2 > ancho / 2:
                    derecha.append((x1, y1, x2, y2))

    # Promedia líneas para suavizar la trayectoria
    def promediar_lineas(lineas):
        if not lineas:
            return None
        x = []
        y = []
        for x1, y1, x2, y2 in lineas:
            x += [x1, x2]
            y += [y1, y2]
        poly = np.polyfit(y, x, 1)
        y1, y2 = altura, int(altura * 0.6)
        x1, x2 = int(np.polyval(poly, y1)), int(np.polyval(poly, y2))
        return x1, y1, x2, y2

    # Dibuja líneas promediadas sobre la imagen
    linea_izquierda = promediar_lineas(izquierda)
    linea_derecha = promediar_lineas(derecha)

    if linea_izquierda:
        # Rojo brillante (más intenso)
        cv2.line(imagen_lineas, (linea_izquierda[0], linea_izquierda[1]), (linea_izquierda[2], linea_izquierda[3]), (0, 0, 255), 4)
    if linea_derecha:
        # Verde neón brillante
        cv2.line(imagen_lineas, (linea_derecha[0], linea_derecha[1]), (linea_derecha[2], linea_derecha[3]), (0, 0, 255), 4)

    # Combina las líneas con el frame original
    resultado = cv2.addWeighted(frame, 1, imagen_lineas, 1, 0)

    if guardar_imagenes:
        cv2.imwrite(os.path.join(carpeta_salida, "resultado.jpg"), resultado)

    return resultado

# Llamadas de ejemplo (ajustar las rutas a tus archivos)
procesar_video("ruta_1.mp4", "resultado_ruta_1.mp4")
procesar_video("ruta_2.mp4", "resultado_ruta_2.mp4")
