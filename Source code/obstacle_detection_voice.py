import cv2
import math
import pyttsx3
# Threshold to detect object
thres = 0.45

# Minimum confidence threshold for object detection
confThreshold = 0.5

# Non-maximum suppression threshold
nmsThreshold = 0.2

# Width and height of the input image
width = 640
height = 480

# Focal length of the camera
focal_length = 700

# Known width of the object to detect (in centimeters)
known_width = 10.0

engine=pyttsx3.init()
rate=engine.getProperty('rate')
engine.setProperty('rate',rate-10)
classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def get_distance(known_width, focal_length, pixel_width):
    """
    Calculate the distance of an object from the camera using the known width
    of the object, the focal length of the camera, and the width of the object
    in pixels in the image.
    """
    return (known_width * focal_length) / pixel_width

def getObjects(img, draw=True, objects=[]):
    try:
        classIds, confs, bbox = net.detect(img, confThreshold=confThreshold, nmsThreshold=nmsThreshold)
        if len(objects) == 0:
            objects = classNames
        objectInfo = []
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1]
                if className in objects:
                    pixel_width = box[2] - box[0]
                    distance = get_distance(known_width, focal_length, pixel_width)
                    objectInfo.append([box, className, distance])
                    distance=distance*0.0254
                    if draw:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, f"Distance:{round(distance, 2)} m", (box[0] + 10, box[1] + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                        engine.say(f"{className} Detected")
                        engine.runAndWait()
        return img, objectInfo
    except:
        pass

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    while True:
        try:
            success,img = cap.read()
            result, objectInfo = getObjects(img)
            cv2.imshow("Obstacle Detection", img)
        except:
            pass

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
