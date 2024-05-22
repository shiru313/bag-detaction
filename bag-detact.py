import cv2
import os
import numpy as np
from tkinter import Tk
import winsound
from playsound import playsound
import time

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects using YOLO
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  # Use numpy's argmax
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

class BagDetection:
    def __init__(self, master):
        self.master = master
        self.master.title("Bag Detection")
        self.net, self.classes, self.output_layers = load_yolo()
        self.detect_bags()
        self.master.destroy()

    def detect_bags(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect objects (e.g., bags)
            class_ids, confidences, boxes = detect_objects(frame, self.net, self.output_layers)
            for i in range(len(boxes)):
                if str(self.classes[class_ids[i]]) in ['backpack', 'handbag']:
                    print("Warning: Person detected with a bag")
                    winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
                    time.sleep(1)
                    


        video_capture.release()

if __name__ == "__main__":
    root = Tk()
    app = BagDetection(root)
    root.mainloop()
