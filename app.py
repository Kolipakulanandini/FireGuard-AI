from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
import threading
import time
import subprocess
import imageio_ffmpeg


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

model = YOLO("model/best.pt")


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "static/uploads"
VIDEO_FOLDER = "static/videos"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# FILE TYPES
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


# ============================================================
# AI SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

FIRE_CONFIDENCE_THRESHOLD = 0.60

SMOKE_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_IMAGE_SIZE = 320

AI_INTERVAL = 0.5


# ============================================================
# FIRE SEVERITY SETTINGS
# ============================================================

WARNING_AREA_PERCENT = 0.5

EMERGENCY_AREA_PERCENT = 3.0

FIRE_CONFIRMATIONS = 2

SAFE_CONFIRMATIONS = 2

WARNING_HOLD_SECONDS = 4.0


# ============================================================
# CAMERA LOCATIONS
# ============================================================

CAMERA_LOCATIONS = {

    "CAM-01": {
        "camera_id": "CAM-01",
        "building": "Narayana Engineering College",
        "floor": "1st Floor",
        "zone": "Computer Centre 2",
        "fire_station": "Nellore Fire Station",
        "route":
            "Computer Centre 2 → Corridor → "
            "Staircase A → Ground Floor → "
            "Main Exit → Safe Area"
    },

    "CAM-02": {
        "camera_id": "CAM-02",
        "building": "Narayana Engineering College",
        "floor": "1st Floor",
        "zone": "Computer Centre 3",
        "fire_station": "Nellore Fire Station",
        "route":
            "Computer Centre 3 → Corridor → "
            "Staircase B → Ground Floor → "
            "Emergency Exit B → Safe Area"
    },

    "CAM-03": {
        "camera_id": "CAM-03",
        "building": "Narayana Engineering College",
        "floor": "2nd Floor",
        "zone": "Computer Lab 1",
        "fire_station": "Nellore Fire Station",
        "route":
            "Computer Lab 1 → Corridor → "
            "Staircase A → 1st Floor → "
            "Ground Floor → Main Exit → Safe Area"
    }
}


active_camera_id = "CAM-01"


# ============================================================
# CAMERA GLOBAL VARIABLES
# ============================================================

camera_running = False

camera_capture_thread = None

camera_ai_thread = None

camera = None

latest_camera_frame = None

latest_ai_frame = None

latest_ai_status = "SAFE"

latest_ai_detections = []

latest_fire_area = 0.0

latest_fire_confidence = 0.0

latest_fire_count = 0

latest_smoke_count = 0

fire_positive_count = 0

safe_count = 0

last_warning_time = 0.0


# ============================================================
# LOCKS
# ============================================================

camera_lock = threading.Lock()

ai_lock = threading.Lock()

state_lock = threading.Lock()


# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE_PATH = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    FACE_CASCADE_PATH
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def allowed_file(filename, allowed_extensions):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in allowed_extensions
    )


# ============================================================
# CALCULATE FIRE BOX AREA
# ============================================================

def calculate_box_area_percent(
    box,
    frame_width,
    frame_height
):

    x1, y1, x2, y2 = box

    box_width = max(
        0,
        x2 - x1
    )

    box_height = max(
        0,
        y2 - y1
    )

    box_area = (
        box_width *
        box_height
    )

    frame_area = (
        frame_width *
        frame_height
    )

    if frame_area <= 0:
        return 0.0

    return (
        box_area /
        frame_area
    ) * 100


# ============================================================
# STRONG FACE FALSE-POSITIVE FILTER
# ============================================================

def box_overlaps_face(
    box,
    faces,
    frame_width=320,
    frame_height=320
):

    if faces is None:
        return False

    if len(faces) == 0:
        return False


    x1, y1, x2, y2 = box


    box_width = max(
        1,
        x2 - x1
    )

    box_height = max(
        1,
        y2 - y1
    )


    box_area = (
        box_width *
        box_height
    )


    frame_area = max(
        1,
        frame_width *
        frame_height
    )


    fire_box_area_percent = (
        box_area /
        frame_area
    ) * 100


    for face in faces:

        fx, fy, fw, fh = face

        fx2 = fx + fw
        fy2 = fy + fh


        # ----------------------------------------------------
        # FACE CENTER
        # ----------------------------------------------------

        face_center_x = (
            fx +
            fw / 2
        )

        face_center_y = (
            fy +
            fh / 2
        )


        face_center_inside_fire = (

            x1 <= face_center_x <= x2

            and

            y1 <= face_center_y <= y2
        )


        # ----------------------------------------------------
        # FACE/FIRE OVERLAP
        # ----------------------------------------------------

        overlap_x1 = max(
            x1,
            fx
        )

        overlap_y1 = max(
            y1,
            fy
        )

        overlap_x2 = min(
            x2,
            fx2
        )

        overlap_y2 = min(
            y2,
            fy2
        )


        overlap_width = max(
            0,
            overlap_x2 -
            overlap_x1
        )

        overlap_height = max(
            0,
            overlap_y2 -
            overlap_y1
        )


        overlap_area = (
            overlap_width *
            overlap_height
        )


        face_area = max(
            1,
            fw * fh
        )


        face_overlap_ratio = (
            overlap_area /
            face_area
        )


        # ----------------------------------------------------
        # RULE 1
        # Face center is inside Fire box
        # ----------------------------------------------------

        if face_center_inside_fire:

            print(
                "Ignored false Fire detection: "
                "face center inside Fire box."
            )

            return True


        # ----------------------------------------------------
        # RULE 2
        # At least 50% of face is inside Fire box
        # ----------------------------------------------------

        if face_overlap_ratio >= 0.50:

            print(
                "Ignored false Fire detection: "
                "large face overlap."
            )

            return True


        # ----------------------------------------------------
        # RULE 3
        # Huge Fire box + partial face overlap
        # ----------------------------------------------------

        if (
            fire_box_area_percent >= 20
            and
            face_overlap_ratio >= 0.20
        ):

            print(
                "Ignored false Fire detection: "
                "large Fire box around face."
            )

            return True


    return False


# ============================================================
# DETECT HUMAN FACES
# ============================================================

def detect_faces(frame):

    try:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_detector.detectMultiScale(

            gray,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(35, 35)
        )


        return faces


    except Exception as e:

        print(
            "Face detection error:",
            e
        )

        return []


# ============================================================
# FIRE SEVERITY ANALYSIS
# ============================================================

def analyze_fire_severity(
    detections,
    frame_width,
    frame_height
):

    global fire_positive_count

    global safe_count

    global last_warning_time


    current_time = time.time()


    fire_found = False

    smoke_found = False

    largest_fire_area = 0.0

    highest_fire_confidence = 0.0


    # ========================================================
    # CHECK ALL VALID DETECTIONS
    # ========================================================

    for detection in detections:

        class_name = (
            detection["class"]
            .lower()
        )

        confidence = (
            detection["confidence"]
        )

        area_percent = (
            detection["area_percent"]
        )


        # ----------------------------------------------------
        # FIRE
        # ----------------------------------------------------

        if class_name == "fire":

            if (
                confidence <
                FIRE_CONFIDENCE_THRESHOLD
            ):

                continue


            fire_found = True


            highest_fire_confidence = max(

                highest_fire_confidence,

                confidence
            )


            largest_fire_area = max(

                largest_fire_area,

                area_percent
            )


        # ----------------------------------------------------
        # SMOKE
        # ----------------------------------------------------

        elif class_name == "smoke":

            if (
                confidence >=
                SMOKE_CONFIDENCE_THRESHOLD
            ):

                smoke_found = True


    # ========================================================
    # FIRE FOUND
    # ========================================================

    if fire_found:

        safe_count = 0

        fire_positive_count += 1

        last_warning_time = current_time


        # ----------------------------------------------------
        # LARGE FIRE
        # ----------------------------------------------------

        if (
            largest_fire_area >=
            EMERGENCY_AREA_PERCENT
        ):

            if (
                fire_positive_count >=
                FIRE_CONFIRMATIONS
            ):

                return (

                    "EMERGENCY",

                    largest_fire_area,

                    highest_fire_confidence
                )


            return (

                "WARNING",

                largest_fire_area,

                highest_fire_confidence
            )


        # ----------------------------------------------------
        # SMALL / MEDIUM FIRE
        # ----------------------------------------------------

        if (
            fire_positive_count >=
            FIRE_CONFIRMATIONS
        ):

            return (

                "WARNING",

                largest_fire_area,

                highest_fire_confidence
            )


        return (

            "SAFE",

            0.0,

            0.0
        )


    # ========================================================
    # SMOKE FOUND
    # ========================================================

    if smoke_found:

        fire_positive_count = 0

        safe_count = 0

        last_warning_time = current_time


        return (

            "WARNING",

            0.0,

            0.0
        )


    # ========================================================
    # NOTHING DETECTED
    # ========================================================

    fire_positive_count = 0

    safe_count += 1


    # --------------------------------------------------------
    # KEEP WARNING FOR A SHORT PERIOD
    # --------------------------------------------------------

    if (
        current_time -
        last_warning_time
        <
        WARNING_HOLD_SECONDS
    ):

        return (

            "WARNING",

            0.0,

            0.0
        )


    # --------------------------------------------------------
    # SAFE
    # --------------------------------------------------------

    if (
        safe_count >=
        SAFE_CONFIRMATIONS
    ):

        return (

            "SAFE",

            0.0,

            0.0
        )


    return (

        "SAFE",

        0.0,

        0.0
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return redirect(
        url_for("camera_page")
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_image():

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "No image selected."


    if not allowed_file(
        file.filename,
        ALLOWED_IMAGE_EXTENSIONS
    ):

        return "Invalid image format."


    filename = secure_filename(
        file.filename
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(filepath)


    results = model(
        filepath,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    fire_count = 0

    smoke_count = 0

    highest_confidence = 0.0


    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            highest_confidence = max(
                highest_confidence,
                confidence
            )


            class_id = int(
                box.cls[0]
            )


            class_name = (
                result.names[class_id]
                .lower()
            )


            if class_name == "fire":

                fire_count += 1


            elif class_name == "smoke":

                smoke_count += 1


    if fire_count > 0:

        status = "FIRE DETECTED"


    elif smoke_count > 0:

        status = "SMOKE DETECTED"


    else:

        status = "SAFE"


    return render_template(

        "index.html",

        image_path=filepath,

        fire_count=fire_count,

        smoke_count=smoke_count,

        highest_confidence=round(
            highest_confidence * 100,
            2
        ),

        status=status
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

@app.route(
    "/upload_video",
    methods=["POST"]
)
def upload_video():

    if "video" not in request.files:

        return "No video uploaded."


    file = request.files["video"]


    if file.filename == "":

        return "No video selected."


    if not allowed_file(
        file.filename,
        ALLOWED_VIDEO_EXTENSIONS
    ):

        return "Invalid video format."


    filename = secure_filename(
        file.filename
    )


    original_path = os.path.join(
        VIDEO_FOLDER,
        filename
    )


    file.save(original_path)


    cap = cv2.VideoCapture(
        original_path
    )


    if not cap.isOpened():

        return "Could not open video."


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 25


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    temp_output = os.path.join(
        VIDEO_FOLDER,
        "temp_" + filename
    )


    final_output = os.path.join(
        VIDEO_FOLDER,
        "processed_" + filename.rsplit(
            ".",
            1
        )[0] + ".mp4"
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    out = cv2.VideoWriter(

        temp_output,

        fourcc,

        fps,

        (width, height)
    )


    fire_count = 0

    smoke_count = 0

    highest_confidence = 0.0

    frame_number = 0

    FRAME_SKIP = 3


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_number += 1


        if (
            frame_number %
            FRAME_SKIP
            == 0
        ):

            results = model(

                frame,

                imgsz=640,

                conf=CONFIDENCE_THRESHOLD,

                verbose=False
            )


            for result in results:

                frame = result.plot()


                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )


                    highest_confidence = max(

                        highest_confidence,

                        confidence
                    )


                    class_id = int(
                        box.cls[0]
                    )


                    class_name = (
                        result.names[class_id]
                        .lower()
                    )


                    if class_name == "fire":

                        fire_count += 1


                    elif class_name == "smoke":

                        smoke_count += 1


        out.write(frame)


    cap.release()

    out.release()


    try:

        ffmpeg_path = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )


        command = [

            ffmpeg_path,

            "-y",

            "-i",

            temp_output,

            "-c:v",

            "libx264",

            "-pix_fmt",

            "yuv420p",

            "-movflags",

            "+faststart",

            "-an",

            final_output
        ]


        subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            check=True
        )


        if os.path.exists(
            temp_output
        ):

            os.remove(
                temp_output
            )


    except Exception as e:

        print(
            "FFmpeg conversion error:",
            e
        )

        final_output = temp_output


    if fire_count > 0:

        status = "FIRE DETECTED"


    elif smoke_count > 0:

        status = "SMOKE DETECTED"


    else:

        status = "SAFE"


    return render_template(

        "index.html",

        video_path=final_output,

        fire_count=fire_count,

        smoke_count=smoke_count,

        highest_confidence=round(
            highest_confidence * 100,
            2
        ),

        status=status
    )


# ============================================================
# CAMERA CAPTURE WORKER
# ============================================================

def camera_capture_worker():

    global camera_running

    global camera

    global latest_camera_frame


    print(
        "Camera capture thread started."
    )


    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )


    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        camera_running = False

        return


    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    print(
        "Camera opened successfully."
    )


    while camera_running:

        ret, frame = camera.read()


        if not ret:

            time.sleep(
                0.05
            )

            continue


        with camera_lock:

            latest_camera_frame = (
                frame.copy()
            )


        time.sleep(
            0.01
        )


    camera.release()

    camera = None

    print(
        "Camera capture thread stopped."
    )


# ============================================================
# AI CAMERA WORKER
# ============================================================

def camera_ai_worker():

    global latest_ai_frame

    global latest_ai_status

    global latest_ai_detections

    global latest_fire_area

    global latest_fire_confidence

    global latest_fire_count

    global latest_smoke_count


    print(
        "AI camera thread started."
    )


    while camera_running:

        # ----------------------------------------------------
        # GET CAMERA FRAME
        # ----------------------------------------------------

        with camera_lock:

            if latest_camera_frame is None:

                frame = None

            else:

                frame = (
                    latest_camera_frame
                    .copy()
                )


        if frame is None:

            time.sleep(
                0.1
            )

            continue


        # ----------------------------------------------------
        # RESIZE FOR AI
        # ----------------------------------------------------

        ai_frame = cv2.resize(

            frame,

            (
                CAMERA_IMAGE_SIZE,
                CAMERA_IMAGE_SIZE
            )
        )


        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        faces = detect_faces(
            ai_frame
        )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        try:

            results = model(

                ai_frame,

                imgsz=320,

                conf=CONFIDENCE_THRESHOLD,

                verbose=False
            )


        except Exception as e:

            print(
                "AI error:",
                e
            )

            time.sleep(
                AI_INTERVAL
            )

            continue


        # ----------------------------------------------------
        # READ DETECTIONS
        # ----------------------------------------------------

        detections = []


        frame_height, frame_width = (
            ai_frame.shape[:2]
        )


        for result in results:

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )


                class_id = int(
                    box.cls[0]
                )


                class_name = (
                    result.names[class_id]
                )


                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )


                # ------------------------------------------------
                # STRONG FACE FILTER
                # ------------------------------------------------

                if (

                    class_name.lower()
                    == "fire"

                    and

                    box_overlaps_face(

                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        faces,

                        frame_width,

                        frame_height
                    )
                ):

                    print(
                        "Ignored false Fire detection "
                        "around human face."
                    )

                    continue


                # ------------------------------------------------
                # FIRE AREA
                # ------------------------------------------------

                area_percent = (
                    calculate_box_area_percent(

                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        frame_width,

                        frame_height
                    )
                )


                # ------------------------------------------------
                # STORE DETECTION
                # ------------------------------------------------

                detections.append({

                    "class":
                        class_name,

                    "confidence":
                        confidence,

                    "area_percent":
                        area_percent,

                    "box":
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                })


        # ----------------------------------------------------
        # COUNT VALID DETECTIONS
        # ----------------------------------------------------

        valid_fire_count = 0

        valid_smoke_count = 0


        for detection in detections:

            class_name = (
                detection["class"]
                .lower()
            )

            confidence = (
                detection["confidence"]
            )


            if (

                class_name == "fire"

                and

                confidence >=
                FIRE_CONFIDENCE_THRESHOLD

            ):

                valid_fire_count += 1


            elif (

                class_name == "smoke"

                and

                confidence >=
                SMOKE_CONFIDENCE_THRESHOLD

            ):

                valid_smoke_count += 1


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        (
            status,
            fire_area,
            fire_confidence

        ) = analyze_fire_severity(

            detections,

            frame_width,

            frame_height
        )


        # ----------------------------------------------------
        # SAVE AI INFORMATION
        # ----------------------------------------------------

        with ai_lock:

            latest_ai_frame = (
                ai_frame.copy()
            )

            latest_ai_status = (
                status
            )

            latest_ai_detections = (
                detections
            )

            latest_fire_area = (
                fire_area
            )

            latest_fire_confidence = (
                fire_confidence
            )

            latest_fire_count = (
                valid_fire_count
            )

            latest_smoke_count = (
                valid_smoke_count
            )


        time.sleep(
            AI_INTERVAL
        )


    print(
        "AI camera thread stopped."
    )


# ============================================================
# CAMERA PAGE
# ============================================================

@app.route("/camera")
def camera_page():

    global camera_running

    global camera_capture_thread

    global camera_ai_thread


    if not camera_running:

        camera_running = True


        camera_capture_thread = (
            threading.Thread(

                target=
                    camera_capture_worker,

                daemon=True
            )
        )


        camera_ai_thread = (
            threading.Thread(

                target=
                    camera_ai_worker,

                daemon=True
            )
        )


        camera_capture_thread.start()

        camera_ai_thread.start()


    with state_lock:

        selected_camera = (
            CAMERA_LOCATIONS[
                active_camera_id
            ]
        )


    return render_template(

        "camera.html",

        camera_location=
            selected_camera,

        camera_options=
            CAMERA_LOCATIONS,

        active_camera_id=
            active_camera_id
    )


# ============================================================
# SELECT ACTIVE CAMERA
# ============================================================

@app.route(
    "/set_camera",
    methods=["POST"]
)
def set_camera():

    global active_camera_id

    global fire_positive_count

    global safe_count

    global last_warning_time

    global latest_ai_status

    global latest_ai_detections

    global latest_fire_area

    global latest_fire_confidence

    global latest_fire_count

    global latest_smoke_count


    data = (
        request.get_json(
            silent=True
        )
        or {}
    )


    requested_camera = (
        data.get(
            "camera_id"
        )
    )


    if (
        requested_camera
        not in CAMERA_LOCATIONS
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Invalid camera ID."
        }), 400


    with state_lock:

        active_camera_id = (
            requested_camera
        )


    # --------------------------------------------------------
    # RESET DETECTION STATE
    # --------------------------------------------------------

    with ai_lock:

        fire_positive_count = 0

        safe_count = 0

        last_warning_time = 0.0

        latest_ai_status = "SAFE"

        latest_ai_detections = []

        latest_fire_area = 0.0

        latest_fire_confidence = 0.0

        latest_fire_count = 0

        latest_smoke_count = 0


    return jsonify({

        "success":
            True,

        "camera":
            CAMERA_LOCATIONS[
                requested_camera
            ]
    })


# ============================================================
# CAMERA STATUS API
# ============================================================

@app.route(
    "/camera_status"
)
def camera_status():

    with state_lock:

        selected_camera = (
            CAMERA_LOCATIONS[
                active_camera_id
            ]
        )


    with ai_lock:

        status = (
            latest_ai_status
        )

        fire_area = (
            latest_fire_area
        )

        fire_confidence = (
            latest_fire_confidence
        )

        fire_count = (
            latest_fire_count
        )

        smoke_count = (
            latest_smoke_count
        )

        detections = (
            latest_ai_detections.copy()
        )


    emergency_active = (
        status ==
        "EMERGENCY"
    )


    return jsonify({

        "status":
            status,

        "camera_running":
            camera_running,

        "fire_count":
            fire_count,

        "smoke_count":
            smoke_count,

        "fire_area":
            round(
                fire_area,
                2
            ),

        "confidence":
            round(
                fire_confidence * 100,
                2
            ),

        "detections":
            detections,

        "camera_id":
            selected_camera[
                "camera_id"
            ],

        "building":
            selected_camera[
                "building"
            ],

        "floor":
            selected_camera[
                "floor"
            ],

        "zone":
            selected_camera[
                "zone"
            ],

        "fire_station":
            selected_camera[
                "fire_station"
            ],

        "route":
            selected_camera[
                "route"
            ],

        "emergency_alert": {

            "active":
                emergency_active,

            "camera_id":
                selected_camera[
                    "camera_id"
                ],

            "building":
                selected_camera[
                    "building"
                ],

            "floor":
                selected_camera[
                    "floor"
                ],

            "zone":
                selected_camera[
                    "zone"
                ],

            "fire_station":
                selected_camera[
                    "fire_station"
                ]
        }
    })


# ============================================================
# LIVE CAMERA FRAME GENERATOR
# ============================================================

def generate_camera_frames():

    global camera_running


    while camera_running:

        # ----------------------------------------------------
        # GET LATEST CAMERA FRAME
        # ----------------------------------------------------

        with camera_lock:

            if latest_camera_frame is None:

                frame = None

            else:

                frame = (
                    latest_camera_frame
                    .copy()
                )


        if frame is None:

            time.sleep(
                0.05
            )

            continue


        # ----------------------------------------------------
        # GET AI DATA
        # ----------------------------------------------------

        with ai_lock:

            status = (
                latest_ai_status
            )

            detections = list(
                latest_ai_detections
            )

            fire_area = (
                latest_fire_area
            )

            fire_confidence = (
                latest_fire_confidence
            )


        frame_height, frame_width = (
            frame.shape[:2]
        )


        # ----------------------------------------------------
        # DRAW DETECTION BOXES
        # ----------------------------------------------------

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["box"]
            )


            scale_x = (
                frame_width /
                CAMERA_IMAGE_SIZE
            )

            scale_y = (
                frame_height /
                CAMERA_IMAGE_SIZE
            )


            x1 = int(
                x1 * scale_x
            )

            y1 = int(
                y1 * scale_y
            )

            x2 = int(
                x2 * scale_x
            )

            y2 = int(
                y2 * scale_y
            )


            class_name = (
                detection["class"]
            )

            confidence = (
                detection["confidence"]
            )

            area_percent = (
                detection["area_percent"]
            )


            label = (

                f"{class_name.upper()} "

                f"{confidence * 100:.1f}% "

                f"| Area "

                f"{area_percent:.2f}%"
            )


            if (
                class_name.lower()
                == "fire"
            ):

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    255
                )


            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                box_color,

                2
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                box_color,

                2
            )


        # ----------------------------------------------------
        # STATUS BANNER
        # ----------------------------------------------------

        if (
            status ==
            "EMERGENCY"
        ):

            banner_text = (
                "EMERGENCY - FIRE DETECTED"
            )

            banner_color = (
                0,
                0,
                255
            )


        elif (
            status ==
            "WARNING"
        ):

            banner_text = (
                "WARNING - CHECK AREA"
            )

            banner_color = (
                0,
                165,
                255
            )


        else:

            banner_text = (
                "SAFE - NO FIRE DETECTED"
            )

            banner_color = (
                0,
                180,
                0
            )


        cv2.rectangle(

            frame,

            (0, 0),

            (
                frame_width,
                55
            ),

            banner_color,

            -1
        )


        cv2.putText(

            frame,

            banner_text,

            (15, 36),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (
                255,
                255,
                255
            ),

            2
        )


        # ----------------------------------------------------
        # FIRE INFORMATION
        # ----------------------------------------------------

        if fire_confidence > 0:

            info_text = (

                "Fire confidence: "

                f"{fire_confidence * 100:.1f}%"
            )


            cv2.putText(

                frame,

                info_text,

                (
                    10,
                    frame_height - 45
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    255,
                    255,
                    255
                ),

                2
            )


            area_text = (

                "Fire area: "

                f"{fire_area:.2f}%"
            )


            cv2.putText(

                frame,

                area_text,

                (
                    10,
                    frame_height - 20
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    255,
                    255,
                    255
                ),

                2
            )


        # ----------------------------------------------------
        # JPEG ENCODING
        # ----------------------------------------------------

        success, buffer = cv2.imencode(

            ".jpg",

            frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                65
            ]
        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"
        )


        time.sleep(
            0.03
        )


# ============================================================
# CAMERA FEED ROUTE
# ============================================================

@app.route(
    "/camera_feed"
)
def camera_feed():

    return Response(

        generate_camera_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# STOP CAMERA
# ============================================================

@app.route(
    "/stop_camera",
    methods=["POST"]
)
def stop_camera():

    global camera_running

    camera_running = False


    return jsonify({

        "success":
            True
    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "      FIREGUARD AI - FIRE & SMOKE DETECTION"
    )

    print(
        "=" * 60
    )

    print(
        "Starting Flask server..."
    )


    app.run(

        debug=False,

        host="127.0.0.1",

        port=5000,

        threaded=True
    )