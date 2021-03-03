import cv2

thres = 0.45 # Threshold to detect object

#cap = cv2.VideoCapture(1)
#cap.set(3,640)
#cap.set(4,480)
#cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img,thres,nms,draw=True,objects=[]):
    #success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)

    if len(objects)==0: objects=classNames
    objectInfo=[]
    returning=[]
    
    valx=0
    valy=0
    valx1=0
    valy1=0
    dif1=0
    dif2=0
    distance=0
    perc_width=0
    ratio=0

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]

            if className in objects:
                objectInfo.append([box,className])
                returning=box
                if (draw):
                    cv2.rectangle(img,box,color=(255,255,224),thickness=1)
                    # cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                    #         cv2.FONT_HERSHEY_COMPLEX,1,(255,255,224),1)
                    #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                          #  cv2.FONT_HERSHEY_COMPLEX,1,(255,255,224),1)

                    valx1=box[0]
                    print('x1:'+str(box[0]))
                    valy1=box[1]
                    print('y1:'+str(box[1]))
                    valx2=box[2]
                    print('x2:'+str(box[2]))
                    valy2=box[3]
                    print('y2:'+str(box[3]))

                    
                    cv2.circle(img, (valx1, valy1), radius=5, color=(0, 0, 255), thickness=-1)
                    
                    cv2.circle(img, (valx1+ valx2, valy1+ valy2), radius=5, color=(0, 0, 255), thickness=-1)
                    
                    perc_width=valx2
                    print('Percieved width:'+str(perc_width))

                    ratio=round((640*480)/(valx2*valy2))
                    distance= round((ratio*1.1529)+17.424,2)
                    #distance= round((1*pow(10,-5)*pow(ratio,4))-(8*pow(10,-5)*pow(ratio,3))-(0.0554*pow(ratio,2))+(2.9542*ratio)+(6.6302),2)

                    print('Distance:'+str(distance))
                    
                    
                    
                   

                    cv2.putText(img,str('Distance:'+str(distance)),(box[0]+100,box[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,224),0)
                   
                   

                    dif1= valx1 + round((valx2)/2)
                    dif2= valy1 + round((valy2)/2)

                    print('dif1:'+str(dif1))
                    print('dif2:'+str(dif2))
                    
                    #print(dif)
                    #print(dif1)
                
                    cv2.circle(img, (dif1, dif2), radius=5, color=(0, 220, 255), thickness=-1)


    return img,objectInfo,returning

  

