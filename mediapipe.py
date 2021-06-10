'''
# MEDIAPIPE
# OPENCV

# DETECCION DE MANOS Y ROSTRO EN IMAGENES Y VIDEO

'''


import cv2
from cv2 import circle as punto
import mediapipe as mp

# ORIGENES DE DATOS
# imgsource es para estáticas de manos
# secondaryImgSrc es para estáticas de rostros

imgsource = "1.jpg"
secondaryImgSrc = "cara.jpg"



mp_dibujo = mp.solutions.drawing_utils
mp_manos = mp.solutions.hands
mp_cara = mp.solutions.face_mesh
video = cv2.VideoCapture(0)

# EL COLOR ES BGR
COLOR_AZ = (240,188,130)
COLOR_RA =(240,204,230)

# Cantidad de Manos
maximoDeManos = int(2)

RADIO_CIRC = int(5)
Modo = "Imagen"

class Mano:
    # 4
    dedoPulgar = mp_manos.HandLandmark.THUMB_TIP
    # 8
    dedoIndice = mp_manos.HandLandmark.INDEX_FINGER_TIP
    # 12
    dedoMedio = mp_manos.HandLandmark.MIDDLE_FINGER_TIP
    # 16
    dedoAnular = mp_manos.HandLandmark.RING_FINGER_TIP
    # 20
    dedoMenique = mp_manos.HandLandmark.PINKY_TIP
    conexiones = mp_manos.HAND_CONNECTIONS
    

def FuncionAEjecutar(argumento, source):
    global Funcion
    global resultado
    Funcion = argumento
    Funcion = Funcion.lower()
    if Funcion=="renderhandlandmarks":  
        if resultado.multi_hand_landmarks is not None:
            for puntoDeReferencia in resultado.multi_hand_landmarks:
                mp_dibujo.draw_landmarks(
                    source, puntoDeReferencia, Mano.conexiones,
                mp_dibujo.DrawingSpec(color=COLOR_RA, thickness=4, circle_radius=RADIO_CIRC),
                mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=4)
                )
    elif Funcion=="coordenadasclave":
        if resultado.multi_hand_landmarks is not None:
            for puntoDeReferencia in resultado.multi_hand_landmarks:
                x1 = int(puntoDeReferencia.landmark[Mano.dedoPulgar].x * width)
                y1 = int(puntoDeReferencia.landmark[Mano.dedoPulgar].y * height)

                x2 = int(puntoDeReferencia.landmark[Mano.dedoIndice].x * width)
                y2 = int(puntoDeReferencia.landmark[Mano.dedoIndice].y * height)

                x3 = int(puntoDeReferencia.landmark[Mano.dedoMedio].x * width)
                y3 = int(puntoDeReferencia.landmark[Mano.dedoMedio].y * height)

                x4 = int(puntoDeReferencia.landmark[Mano.dedoAnular].x * width)
                y4 = int(puntoDeReferencia.landmark[Mano.dedoAnular].y * height)

                x5 = int(puntoDeReferencia.landmark[Mano.dedoMenique].x * width)
                y5 = int(puntoDeReferencia.landmark[Mano.dedoMenique].y * height)

                punto(imagen, (x1,y1), RADIO_CIRC, COLOR_AZ, -1)
                punto(imagen, (x2,y2), RADIO_CIRC, COLOR_AZ, -1)
                punto(imagen, (x3,y3), RADIO_CIRC, COLOR_AZ, -1)
                punto(imagen, (x4,y4), RADIO_CIRC, COLOR_AZ, -1)
                punto(imagen, (x5,y5), RADIO_CIRC, COLOR_AZ, -1)
    elif Funcion=="verhandlandmarks":
        if resultado.multi_hand_landmarks is not None:
            for puntoDeReferencia in resultado.multi_hand_landmarks:
                print(puntoDeReferencia)
    elif Funcion=="verhandedness":
        print(resultado.multi_handedness)
    elif Funcion=="verlandmarks":
        print(resultado.multi_hand_landmarks)
    else:
        print("NADA especificado que sea utilizable")
    

def render(argh):
    global resultado
    global imagen
    global maximoDeManos
    global width
    global height
    with mp_manos.Hands(
        static_image_mode=True,
        max_num_hands=maximoDeManos,
        min_detection_confidence=0.5
        ) as manos:

        imagen = cv2.imread(imgsource)
        height, width, _ = imagen.shape
        imagen = cv2.flip(imagen, 1)

        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = manos.process(imagen_rgb)

        FuncionAEjecutar(argh, imagen)

        imagen = cv2.flip(imagen, 1)
    cv2.imshow("Image", imagen)

def renderVideo():
    # funcion modo OK
    global Modo
    global resultado
    Modo = "Video"
    print("Presione ESC para salir :)")
    with mp_manos.Hands(
        static_image_mode=False,
        max_num_hands=maximoDeManos,
        min_detection_confidence=0.5) as manos:

        while True:
            ret, frame = video.read()
            if ret == False:
                break

            height, width, _ = frame.shape
            frame = cv2.flip(frame,1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resultado = manos.process(frame_rgb)
            print("Handedness:", resultado.multi_handedness)
            FuncionAEjecutar("renderHandLandmarks", frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

def renderRostro():
    with mp_cara.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_tracking_confidence=0.5,
        ) as cara:

        imagen = cv2.imread(secondaryImgSrc)
        height, width, _ = imagen.shape
        image_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = cara.process(image_rgb)
        print("Face landmarks: ", resultado.multi_face_landmarks)

        if resultado.multi_face_landmarks is not None:
            for flanders in resultado.multi_face_landmarks:
                mp_dibujo.draw_landmarks(imagen, flanders,
                mp_cara.FACE_CONNECTIONS,
                mp_dibujo.DrawingSpec(color=COLOR_RA, thickness=1, circle_radius=RADIO_CIRC),
                mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1))
                
                
        
        cv2.imshow("Imagen", imagen)

def renderVideoRostro():
    with mp_cara.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_tracking_confidence=0.5,
        ) as cara:

        while True:
            ret, frame = video.read()
            if ret == False:
                break
            frame = cv2.flip(frame,1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = cara.process(frame_rgb)

            if resultado.multi_face_landmarks is not None:
                for flanders in resultado.multi_face_landmarks:
                    mp_dibujo.draw_landmarks(frame, flanders,
                    mp_cara.FACE_CONNECTIONS,
                    mp_dibujo.DrawingSpec(color=COLOR_RA, thickness=1, circle_radius=RADIO_CIRC),
                    mp_dibujo.DrawingSpec(color=COLOR_AZ, thickness=1))

            cv2.imshow("Cara", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
    

#print(Modo)
render("renderHandLandmarks")
#print(Modo)


if Modo == "Imagen":  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif Modo == "Video":
    video.release()
    cv2.destroyAllWindows()
