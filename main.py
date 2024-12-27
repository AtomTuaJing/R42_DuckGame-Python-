import cv2
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

model = YOLO("best.pt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    

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
                    await websocket.send_text(str(centerY))
                    print(centerY)
                else:
                    await websocket.send_text("CenterY Not Found")

    cap.release()
    cv2.destroyAllWindows()