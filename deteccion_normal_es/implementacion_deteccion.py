#Importacion de las librerias necesarias
import cv2 #manipular imagenes
import time #latencia
import requests
from ultralytics import YOLO #Modelos a usar

#Instanciar modelo
model = YOLO('yolo11n.pt')

#Inicialziar la camar
cap = cv2.VideoCapture(0)

#configurar la camara para verla en pantalla compelta
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

intruso = None

def send_message(msg:str):
    wehbook_url = 'https://discord.com/api/webhooks/1430551427595374615/HCDKg0yR2GafgvK6VK21CrmW81mnhI2LpoLPdX9yCPj--Paj6iad4VQvGzzGbxVyWf2G'
    data = {"content": msg}
    requests.post(wehbook_url, json=data)

if not cap.isOpened():
    print('No se pudo abrir la camara')
else:
    while True:
        #Extraer los frames de la camara
        ret, frame = cap.read()
        #Si no está guardando ningun frame cortar
        if not ret:
            break

        #Inicialziar tiempo para oder revisar la altencia (Tiempo ue se tarda en procesar los fps)
        start_time = time.time()

        #Pasarle al modelo el frame para que empiece y nos detecte 
        result = model(
            frame,
            conf=0.7, #Solamente me detecta aquello con 70% de confianza
            classes=[0] #solamente me detecta las clases [0] personas
        )

        #Calcular cuanto se demoró desde que inció el tiempo hasta que proesó el frame
        latency = (time.time() - start_time) * 1000 #Milisegundos

        #Acceder de la lista de los bounding boxes generados el primero 
        bboxes_conf = result[0].boxes

        #Si hay una deteccion es decir el bounding boxes es diferente a none entonces extraer esa ifnroamcion y dibujar sobre el frame
        if bboxes_conf is not None and len(bboxes_conf) > 0:
            #Extraer datos de los bounding boxes
            bboxes = bboxes_conf.xyxy.cpu().numpy() #Extraer las cordenadas de los bounding box detectados mismo tipo de dato y dispositivo par ano generar problemas 
            confs = bboxes_conf.conf.cpu().numpy() #Extaer la confianza de cada bounding box (el score)
            classes = bboxes_conf.cls.cpu().numpy() #Extraer los indices de las clases detectadas

            if intruso is None:
                for i in classes:
                    if i == 0:
                        send_message("Se ha detectado un intruso, Por favor revise su negocio ")
                        intruso = 1

            #Iterar por cada uno de los bounding boxes detectados para escrbirilos sobre el frame actual
            for i , box in enumerate(bboxes):
                #Extraer cordenadas del bbox
                x1, y1, x2, y2 = map(int, box)

                #Obtener el nombre de la clase a la que pertenece el indice y en caso de que no lo tenga usar el indice como clase
                clase_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))

                #Crear una etiqueta que nos de nombre y confiazna 
                label = f'{clase_name} | {confs[i]:.2f}'

                #Dibujar el rectangulo en el frame (bounding box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255))

                #Dibujar la etiqueta arriba del objeto
                cv2.putText(frame, label,(x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            
        #Mostrar la latenicia en el frame para conocer el rendimiento 
        cv2.putText(frame, f'{latency:.1f}/ms', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        #Mostrar el frame procesado en real time
        cv2.imshow('YOLOv11 - Deteccion En tiempo real',frame)

        #Salir al presionar q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#Con el modelo de yolo se pueden encontrar mas de 80 clases en detecciones