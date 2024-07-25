import cv2
import urllib.request
import numpy as np
import pyodbc
from datetime import datetime
from collections import Counter

# Database connection using Windows authentication
conn = pyodbc.connect('DRIVER={SQL Server};'
                      'SERVER=DESKTOP-SJB3JL3;'
                      'DATABASE=DetectionObject;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()

# Create table if it does not exist
cursor.execute('''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='DetectionObject' and xtype='U')
    CREATE TABLE DetectionObject (
        id INT IDENTITY(1,1) PRIMARY KEY,
        timestamp DATETIME,
        name NVARCHAR(50),
        amount INT
    )
''')
conn.commit()

# Object detection setup
url = 'http://192.168.137.120/cam-hi.jpg'
winName = 'ESP32 CAMERA'
captureWinName = 'Captured Image'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(captureWinName, cv2.WINDOW_AUTOSIZE)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Debug: Print the number of classes loaded
print(f"Number of classes loaded: {len(classNames)}")

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def process_image(img, display_window=None):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    if len(classIds) != 0:
        detected_objects = []
        displayed_objects = set()

        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Debug: Print classId to ensure it's within range
            print(f"Detected classId: {classId}")

            if 1 <= classId <= len(classNames):
                name = classNames[classId - 1]
                detected_objects.append(name)

                # Display each object only once
                if name not in displayed_objects:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                    cv2.putText(img, name, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    displayed_objects.add(name)
            else:
                print(f"Invalid classId: {classId}")

        # Count the number of each detected object in the current frame
        object_counts = Counter(detected_objects)

        # Write data to SQL server if display_window is the captured window
        if display_window == captureWinName:
            timestamp = datetime.now()
            for name, amount in object_counts.items():
                cursor.execute("INSERT INTO DetectionObject (timestamp, name, amount) VALUES (?, ?, ?)",
                               timestamp, name, amount)
                conn.commit()
    
    if display_window:
        cv2.imshow(display_window, img)

captured_image = None  # Global variable to store the captured image

while True:
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    process_image(img, winName)
    
    key = cv2.waitKey(5)
    if key & 0xFF == 27:  # ESC key to break
        break
    elif key & 0xFF == ord('h'):  # 'H' key to capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured_image = img.copy()  # Store the captured image
        cv2.imwrite(f'captured_{timestamp}.jpg', captured_image)
        print(f"Image captured and saved as captured_{timestamp}.jpg")
        cv2.imshow(captureWinName, captured_image)
        process_image(captured_image, captureWinName)

cv2.destroyAllWindows()
conn.close()
