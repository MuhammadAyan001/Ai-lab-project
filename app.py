from flask import Flask, render_template, request, jsonify
import base64, cv2, numpy as np, io, os, uuid
from PIL import Image
import mediapipe as mp
from collections import defaultdict

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, 
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Session memory
sessions = defaultdict(lambda: {"pushup": {"stage": None, "count": 0, "last_angle": None},
                                "pullup": {"stage": None, "count": 0, "last_y": None}})

def decode_base64_image(data_url):
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)[:,:,::-1]
    return arr

def angle_between(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    img_b64 = data["image"]
    exercise = data["exercise"]
    session_id = data.get("session_id") or str(uuid.uuid4())

    img = decode_base64_image(img_b64)
    h, w, _ = img.shape

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    response = {"session_id": session_id, "found": False, "exercise": exercise,
                "angle": None, "count": None, "feedback": "-", "landmarks": []}

    if not results.pose_landmarks:
        return jsonify(response)

    response["found"] = True
    lm = results.pose_landmarks.landmark

    for i, l in enumerate(lm):
        response["landmarks"].append({"index": i, "x": l.x, "y": l.y})

    # side selection
    left_visibility = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
    right_visibility = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
    use_left = left_visibility > right_visibility

    if exercise == "pushup":
        if use_left:
            shoulder = (lm[11].x * w, lm[11].y * h)
            elbow = (lm[13].x * w, lm[13].y * h)
            wrist = (lm[15].x * w, lm[15].y * h)
        else:
            shoulder = (lm[12].x * w, lm[12].y * h)
            elbow = (lm[14].x * w, lm[14].y * h)
            wrist = (lm[16].x * w, lm[16].y * h)

        angle = angle_between(shoulder, elbow, wrist)
        response["angle"] = round(angle, 1)

        state = sessions[session_id]["pushup"]
        if angle < 95:
            state["stage"] = "down"
        if angle > 160 and state["stage"] == "down":
            state["stage"] = "up"
            state["count"] += 1

        response["count"] = state["count"]
        if angle > 170:
            response["feedback"] = "Straight arms — lower more."
        elif angle < 80:
            response["feedback"] = "Good depth — push back up!"
        else:
            response["feedback"] = "Maintain control."

    elif exercise == "pullup":
        if use_left:
            shoulder_y = lm[11].y * h
            hip_y = lm[23].y * h
        else:
            shoulder_y = lm[12].y * h
            hip_y = lm[24].y * h

        delta = hip_y - shoulder_y
        response["angle"] = round(delta, 3)

        state = sessions[session_id]["pullup"]
        if delta > 40:
            state["stage"] = "up"
            state["count"] += 1
        elif delta < 20:
            state["stage"] = "down"

        response["count"] = state["count"]

        if delta < 20:
            response["feedback"] = "Dead hang — start from bottom."
        elif delta > 40:
            response["feedback"] = "Great pull! Return slow."
        else:
            response["feedback"] = "Keep pulling!"

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True) 