#mediapipe
#deteccion cuerpo entero - implementación en imágenes estáticas

#module: https://google.github.io/mediapipe/solutions/holistic

import cv2
import mediapipe as mp
import numpy as np

cv = cv2 # alias workaround para que en caso de usar cv de el mismo resultado

#variables de inicio
mp_dibujo = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#variables varias para simplificar un poco las cosas
mp_manos = mp.solutions.hands
holisticv = mp_holistic.Holistic
abririmagen = cv2.imread
mostrarimagen = cv.imshow
#srcimg = "francordi2.jpg"

srcimg = "proy.jpg"         # imagen estatica
video = cv2.VideoCapture("video.mp4") # video (definir como numero para usar la cámara como source, por ejemplo 0)

colorCv = cv2.cvtColor
colorCvConfig = cv2.COLOR_BGR2RGB

#definiendo algunos colores
COLOR_AM = (0, 255, 255)    # Amarillo
COLOR_AZ = (240,188,130)    # Azul
COLOR_RA = (240,204,230)    # Rojo
COLOR_NA = (0, 128, 255)    # Naranja
COLOR_RO = (128, 0, 255)    # Rosa

# variables varias
RADIO_CIRC = int(4)         # radio de los puntos
ALT_RADIO_CIRC = int(3)     # tamaño alternativo para el radio de los puntos
ESCALA = (468, 702)         # escala para que el output (window) no sea tan grande

def cv_size():              # funcion para retornar el valor original de la imagen por consola
    imaori = abririmagen(srcimg)
    #global srcimg
    return tuple(imaori.shape[1::-1])

ESCALA_ORI = cv_size()

print("Tamaño de imagen original:", ESCALA_ORI, "\nTamaño Establecido:", ESCALA)

# comparar el tamaño definido contra el de la imagen original
if ESCALA < ESCALA_ORI:
    print("El tamaño establecido es menor al de la imagen original")
elif ESCALA > ESCALA_ORI:
    print("El tamaño establecido es mayor al de la imagen original")
elif ESCALA == ESCALA_ORI:
    print("La escala y el tamaño de imagen usado coinciden")

Modo = "Imagen"
MatPlot = "1"       # activar o desactivar gráfica
MatPlotType = "1"   # modo de gráfica (0 = puntos, 1 = puntos conectados)

# static_image_mode (boolean)       # UN ayuda-memoria :)
# model_complexity (0,1,2)
# smooth_landmarks (boolean)
# min_detection_confidence (float)
# min_tracking_confidence (float)

class Mano:
    conexiones = mp_manos.HAND_CONNECTIONS
    #conexiones = mp_holistic.HAND_CONNECTIONS

def RenderMathplotLibCords():       # FUNCION PARA INICIAR EL GRAFICO (MatPlot)
    if MatPlotType == "0":          # Según el tipo definido (MatPlotType)
        mp_dibujo.plot_landmarks(
            resultado.pose_world_landmarks  # render landmarks
        )
    elif MatPlotType == "1":        # en caso de ser segundo tipo
        mp_dibujo.plot_landmarks(
            resultado.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS    # render landmark con conexiones
        )
    else:   # si no es ninguno de los dos tipos, retorna error
        print("La variable MatPlotType no está configurada correctamente.\nSolo se acepta un valor entero.")

def FuncionAEjecutar(inputData, endData):

    # frame -> imagen
    # vframe -> video

    # MANOS
    mp_dibujo.draw_landmarks(
        inputData, endData.left_hand_landmarks, Mano.conexiones,
        mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1, circle_radius=ALT_RADIO_CIRC),
        mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))

    mp_dibujo.draw_landmarks(
        inputData, endData.right_hand_landmarks, Mano.conexiones,
        mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1, circle_radius=ALT_RADIO_CIRC),
        mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))

    # cuerpo
    mp_dibujo.draw_landmarks(
        inputData, endData.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_dibujo.DrawingSpec(color=COLOR_RO, thickness=2, circle_radius=ALT_RADIO_CIRC),
        mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))

def RenderHolistic():
    global resultado
    with holisticv(
        static_image_mode=True,         # imagen ESTATICA
        model_complexity=2) as holistic:

        ima = abririmagen(srcimg)       
        frame = cv2.resize(ima, ESCALA)
        frame_rgb = colorCv(frame, colorCvConfig)   # transforma el color space de RGB a BGR (inverso)

        resultado = holistic.process(frame_rgb)

        # FACE-DETECT
        # FACE_CONNECTIONS ahora se llama FACEMESH_TESSELATION
        mp_dibujo.draw_landmarks(
            frame, resultado.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1, circle_radius=RADIO_CIRC),   # puntos
            mp_dibujo.DrawingSpec(color=COLOR_NA, thickness=1))                             # conexiones
        
        FuncionAEjecutar(frame, resultado)

        mostrarimagen("imagen", frame)                          # muestra imagen con los puntos y conexiones anteriormente definidos

        if MatPlot == "1":                                      # evalua si mostrar o no el render de puntos en una gráfica
            print("Se ha activado la representación gráfica")
            RenderMathplotLibCords()
        elif MatPlot == "0":
            print("No se ha activado la representación gráfica")
        else:
            print("La variable MatPlot no está configurada correctamente.\nSolo se acepta un valor binario.")
            pass

def RenderVideoHolistic():  # futura implementación para video en lugar de imagen
    global results
    global Modo
    Modo = "video"
    with holisticv(
        static_image_mode=True,
        model_complexity=2) as holistic:

        while True:
            ret, vframe = video.read()
            if ret == False:
                break
            frame_rgb = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            FuncionAEjecutar(vframe, results)
            mostrarimagen("Video", vframe)
            if cv2.waitKey(1) & 0xFF == 27:
                break


RenderHolistic()
#RenderVideoHolistic()

# La funcion de Video tiene mucha lentitud en equipos donde TensorFlow ->
# se ejecuta en CPU.

if Modo == "Imagen" or Modo == "imagen":  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif Modo == "Video" or Modo == "Video":
    video.release()
    #print("video no implementado")
    cv2.destroyAllWindows()
