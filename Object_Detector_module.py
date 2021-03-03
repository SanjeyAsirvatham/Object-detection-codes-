from Algorithm4 import *



cap= cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
    #cap.set(10,70)

while True:
    success,img = cap.read()
    result,objectInfo,returning=getObjects(img,0.50,0.1, objects=['laptop'])
    print(objectInfo)
    print(returning)

    cv2.imshow("Output",img)
    cv2.waitKey(1)