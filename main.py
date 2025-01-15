import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

file_path = "/Users/atomtuajing/Documents/duck_game/R42_DuckGame-Flutter-/Flutter/duck_game/assets/data.txt" # fix file path

while True:
        success, frame = cap.read()

        if not success:
            break

        results = model(frame, conf=0.5)

        for box in results[0].boxes:
            if box.conf > 0.5:
                _, y1, _, y2 = box.xyxy[0]
                centerY = (y1 + y2) / 2

                if centerY:
                    with open(file_path, 'w') as file:
                        file.write(str(centerY))
                    print(centerY)
                    
cap.release()
cv2.destroyAllWindows()