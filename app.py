from flask import Flask, render_template, request, Response
import cv2
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from datetime import datetime
import time


app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def deteksi_pose():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    counter_kanan = 0 
    counter_kiri = 0
    stage_kanan = None
    stage_kiri = None

    index = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                mata_kanan_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
                mata_kanan_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
                telinga_kanan = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                
                mata_kiri_inner = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
                mata_kiri_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
                telinga_kiri = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                
                angle_kanan = calculate_angle(mata_kanan_inner, mata_kanan_outer, telinga_kanan)
                angle_kiri = calculate_angle(mata_kiri_inner, mata_kiri_outer, telinga_kiri)

                # cv2.putText(image, str(angle_kanan), 
                #             tuple(np.multiply(mata_kanan_outer, [640, 480]).astype(int)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                #                     )
                
                if angle_kanan > 120:
                    stage_kanan = "diem"

                if angle_kanan < 70 and stage_kanan =='diem':
                    stage_kanan="kanan"
                    counter_kanan +=1

                    time.sleep(0.5)
                    capture, image = cap.read()
                    file_name = f"{index}_Lihat Kanan_{datetime.now().strftime('%H:%M:%S')}.png"
                    cv2.imwrite(file_name, image)
                    index+=1
                    
                if angle_kiri > 120:
                    stage_kiri = "diem"

                if angle_kiri < 70 and stage_kiri =='diem':
                    stage_kiri="kiri"
                    counter_kiri +=1
                    
                    time.sleep(0.5)
                    capture, image = cap.read()
                    file_name = f"{index}_Lihat Kiri_{datetime.now().strftime('%H:%M:%S')}.png"
                    cv2.imwrite(file_name, image)
                    index+=1

            except:
                pass
            
            cv2.putText(image, 'KANAN', (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter_kanan), 
                        (5, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                        
            cv2.putText(image, 'KIRI', (180, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter_kiri), 
                        (180, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/result")
def result():
    result_folder = 'static/images/result'
    image_paths = os.listdir(result_folder)
    len_paths = len(image_paths)
    right = []
    left = []
    for i in range(len_paths):
        split = image_paths[i].split('_')
        if split[1] == "Lihat Kanan":
            right.append(split[1])
        else:
            left.append(split[1])
    return render_template("result.html", image_paths = image_paths, len_paths=len_paths, len_right = len(right), len_left = len(left))

@app.route("/video_feed")
def video_feed():
    return Response(deteksi_pose(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")