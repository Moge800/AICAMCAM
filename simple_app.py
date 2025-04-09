import os
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO


os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
app = Flask(__name__)
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(frame, save=False, conf=0.3, classes=[0], verbose=False)
            frame = results[0].plot()
            counter = results[0].boxes.data.shape[0]
            cv2.putText(frame, f"DETECT_COUNT: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
