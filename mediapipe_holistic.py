#mediapipe
#deteccion cuerpo entero - implementación en imágenes estáticas

#module: https://google.github.io/mediapipe/solutions/holistic

import cv2
import mediapipe as mp
import numpy as np

cv = cv2

mp_dibujo = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_manos = mp.solutions.hands

holisticv = mp_holistic.Holistic
abririmagen = cv2.imread
mostrarimagen = cv.imshow
#srcimg = "francordi2.jpg"
srcimg = "proy.jpg"
colorCv = cv2.cvtColor
colorCvConfig = cv2.COLOR_BGR2RGB

COLOR_AM = (0, 255, 255)
COLOR_AZ = (240,188,130)
COLOR_RA =(240,204,230)
COLOR_NA =(0, 128, 255)
COLOR_RO = (128, 0, 255)

RADIO_CIRC = int(4)
ALT_RADIO_CIRC = int(3)
ESCALA = (468, 702)

def cv_size():
    imaori = abririmagen(srcimg)
    #global srcimg
    return tuple(imaori.shape[1::-1])

ESCALA_ORI = cv_size()

print("Tamaño de imagen original:", ESCALA_ORI, "\nTamaño Establecido:", ESCALA)

if ESCALA < ESCALA_ORI:
    print("El tamaño establecido es menor al de la imagen original")
elif ESCALA > ESCALA_ORI:
    print("El tamaño establecido es mayor al de la imagen original")
elif ESCALA == ESCALA_ORI:
    print("La escala y el tamaño de imagen usado coinciden")

Modo = "Imagen"
MatPlot = "1"
MatPlotType = "1"

# static_image_mode (boolean)
# model_complexity (0,1,2)
# smooth_landmarks (boolean)
# min_detection_confidence (float)
# min_tracking_confidence (float)

class Mano:
    conexiones = mp_manos.HAND_CONNECTIONS
    #conexiones = mp_holistic.HAND_CONNECTIONS

def RenderMathplotLibCords():
    if MatPlotType == "0":
        mp_dibujo.plot_landmarks(
            resultado.pose_world_landmarks
        )
    elif MatPlotType == "1":
        mp_dibujo.plot_landmarks(
            resultado.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    else:
        print("La variable MatPlotType no está configurada correctamente.\nSolo se acepta un valor entero.")

def RenderHolistic():
    global resultado
    with holisticv(
        static_image_mode=True,
        model_complexity=2) as holistic:

        ima = abririmagen(srcimg)
        frame = cv2.resize(ima, ESCALA)
        frame_rgb = colorCv(frame, colorCvConfig)

        resultado = holistic.process(frame_rgb)

        # FACE-DETECT
        # FACE_CONNECTIONS ahora se llama FACEMESH_TESSELATION
        mp_dibujo.draw_landmarks(
            frame, resultado.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1, circle_radius=RADIO_CIRC),
            mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))
        
        # MANOS

        mp_dibujo.draw_landmarks(
            frame, resultado.left_hand_landmarks, Mano.conexiones,
            mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1, circle_radius=ALT_RADIO_CIRC),
            mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))

        mp_dibujo.draw_landmarks(
            frame, resultado.right_hand_landmarks, Mano.conexiones,
            mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1, circle_radius=ALT_RADIO_CIRC),
            mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))

        # cuerpo

        mp_dibujo.draw_landmarks(
            frame, resultado.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_dibujo.DrawingSpec(color=COLOR_RO, thickness=2, circle_radius=ALT_RADIO_CIRC),
            mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))

        mostrarimagen("imagen", frame)

        if MatPlot == "1":
            print("Se ha activado la representación gráfica")
            RenderMathplotLibCords()
        elif MatPlot == "0":
            print("No se ha activado la representación gráfica")
        else:
            print("La variable MatPlot no está configurada correctamente.\nSolo se acepta un valor binario.")
            pass

RenderHolistic()

if Modo == "Imagen":  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif Modo == "Video":
    #video.release()
    print("video no implementado")
    cv2.destroyAllWindows()
