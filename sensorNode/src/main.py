import torch
import numpy as np
import cv2
from time import time
from datetime import datetime
import paho.mqtt.client as mqtt

def on_publish(client,userdata,result):
    print("Published!")

def on_disconnect(client,userdata,result):
    client.loop_stop()

if __name__ == '__main__':

    #Loding the model 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

    #Cuda support if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device:", device)

    #Connecting to mqtt client
    client = mqtt.Client(client_id="sensor")
    client.loop_start()
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    print("Connecting to mqtt broker")
    client.connect("192.168.0.100",1883,60)
    print("connection stablished")
    #Opening the camera
    gst_cap = 'nvarguscamerasrc sensor_mode=0 ! video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12 ! nvvidconv  ! video/x-raw, format=BGRx, width=960, height=616 ! videoconvert ! video/x-raw, format=BGR ! appsink'
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    
    #Rat Discovery information
    discoveryImage = []
    discoveryStart = datetime.now()
    discoverymean = 0.0
    discoveryEnd = datetime.now()

    #Reading the camera frame by frame
    while True:
        ret, frame = cap.read()
        assert ret
        
        #for fps
        start = time()

        # Evaluatio of the frame
        model.to(device)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # turning the image to grayscale for speed!
        filter = np.array([[-1,-1,-1,-1,-1],
                          [-1,2,2,2,-1],
                          [-1,2,8,2,-1],
                          [-1,2,2,2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0 # Guassian filter for edge enhancement
        
        frame = cv2.filter2D(frame, -1, filter) # enhancing the image

        #Prediction
        model.conf = 0.5
        model.iou = 0.25
        results = model(frame)
        lables, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        #Drawing the bounding boxes if necessary(only for the frame that we send to mqtt broker!)
        n = len(lables)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        if n!=0:
            tempsum = 0.0
            tempcount = 0
            for i in range(n):
                row = cord[i]
                tempcount+=1
                tempsum+=row[4].item()
                cv2.rectangle(frame,(int(row[0].item()*x_shape),int(row[1].item()*y_shape)),(int(row[2].item()*x_shape),int(row[3].item()*y_shape)),(0,0,255),2)
                cv2.putText(frame, "Rat: %0.2f" %row[4].item(), (int(row[0].item()*x_shape),int(row[1].item()*y_shape) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            if tempsum/tempcount > discoverymean:
                discoveryImage = frame
                discoverymean = tempsum/tempcount
            discoveryEnd = datetime.now()

            
            
        #Sending the discovery to mqtt broker
        elif discoverymean!=0.0:
            print("Gotcha! I'm gonna send your picture to the master")
            print("connection stablished")
            success, encoded_image = cv2.imencode('.jpeg', discoveryImage)
            
            #convert encoded image to bytearray
            bytarr = encoded_image.tobytes()
            print("sending the message")
            message = str(bytarr,'iso-8859-1') + "StartTime"+ discoveryStart.strftime("%m/%d/%Y, %H:%M:%S") + "EndTime" + discoveryEnd.strftime("%m/%d/%Y, %H:%M:%S")
            client.publish("Rat", message)
            print("message sent")
            print("--------------------------------")
            discoveryStart = datetime.now()
            discoverymean = 0.0

        end = time()
        fps = 1/np.round(end - start, 2)
        cv2.putText(frame,f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        cv2.imshow('yolov5', frame)
        
        if cv2.waitKey(5) & 0xFF ==27:
                break
    
    cap.release()
    cv2.destroyAllWindows()
