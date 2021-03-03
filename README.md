# Object-detection-Algorithm

**Content:**
This is a series of algorithms developed in python and opencv for a company project i have been working on. Hope the code published is resourceful.

The series of codes published are codes adapted from multiple open sources online. The typical source is youtube. When the links for these identified it will be mentioned in this repository at a latter date. 

The following describes the structure of the content published in this folder. 

Algorithm.py is a code which uses tracking algorithm from the inbuilt library from opencv to track an object assigned by the bounding box created.

Algorithm2.py uses a different approach to evaluate to track an object. It uses the HSV values as a filter.

Algorithm3.py is basically an application to evaluate the hsv values for the former algorithm.

Algorithm4.py is the functionally active code with utilises the coco.names and the frozen inference_graph.pb  and the ssd_mobilenet_v3_large to evaluate multiple objects. The code also tries to evaluate the distance of an object from the camera being used by using linear regression. However this is not the accurate approach to evaluate the distance of an object. Stereo vision would be the next best alternative is computer vision was the route to evaluate the distance. 


**Conclusion:**
Further improvements should be made to this code to cater the purpose. 


**Sources:**

