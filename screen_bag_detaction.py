# import cv2
# import numpy as np
# import mss
# from tkinter import Tk

# # Load YOLO model
# def load_yolo():
#     net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#     with open("coco.names", "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     return net, classes, output_layers

# # Detect objects using YOLO
# def detect_objects(img, net, output_layers):
#     height, width, channels = img.shape
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Adjust confidence threshold as needed
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#     return class_ids, confidences, boxes

# class BagDetection:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Bag Detection")
#         self.net, self.classes, self.output_layers = load_yolo()
#         self.detect_bags()
#         self.master.destroy()

#     def detect_bags(self):
#         with mss.mss() as sct:
#             monitor = sct.monitors[1]  # Adjust this if you have multiple monitors

#             while True:
#                 screen_shot = sct.grab(monitor)
#                 frame = np.array(screen_shot)
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

#                 # Detect objects (e.g., bags)
#                 class_ids, confidences, boxes = detect_objects(frame, self.net, self.output_layers)
#                 for i in range(len(boxes)):
#                     if str(self.classes[class_ids[i]]) in ['backpack', 'handbag']:
#                         print("Warning: Person detected with a bag")

#                 # Save the resulting frame to a file (optional)
#                 # cv2.imwrite("frame.jpg", frame)

# if __name__ == "__main__":
#     root = Tk()
#     app = BagDetection(root)
#     root.mainloop()


import cv2
import numpy as np
import mss
from tkinter import Tk
import torch
import winsound
import time


# Load YOLO model (using YOLOv5 for better performance)
def load_yolo():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model
    return model

# Detect objects using YOLO
def detect_objects(img, model):
    results = model(img)  # Perform inference
    return results.xyxy[0]  # Extract detection results

class BagDetection:
    def __init__(self, master):
        self.master = master
        self.master.title("Bag Detection")
        self.model = load_yolo()
        self.detect_bags()
        self.master.destroy()

    def detect_bags(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Adjust this if you have multiple monitors

            while True:
                screen_shot = sct.grab(monitor)
                frame = np.array(screen_shot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Detect objects (e.g., bags)
                detections = detect_objects(frame, self.model)
                for *box, conf, cls in detections:
                    if self.model.names[int(cls)] in ['backpack', 'handbag']:
                        print("Warning: Person detected with a bag")
                        winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
                        time.sleep(1)
                        # Log the detection
                        self.log_detection(self.model.names[int(cls)], conf, box)

    def log_detection(self, object_name, confidence, box):
        with open("detection_log.txt", "a") as f:
            f.write(f"{object_name} detected with {confidence:.2f} confidence at {box}\n")
        print(f"{object_name} detected with {confidence:.2f} confidence at {box}")

if __name__ == "__main__":
    root = Tk()
    app = BagDetection(root)
    root.mainloop()
