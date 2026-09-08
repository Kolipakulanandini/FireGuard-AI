from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os

app = Flask(__name__)

# Secret key for storing temporary results
app.secret_key = "fireguard-secret-key"

# Load Fire + Smoke AI model
model = YOLO("model/best.pt")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed image types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():

    # Get previous detection data
    data = session.pop("detection", None)

    if data:
        return render_template(
            "index.html",
            uploaded_image=data["uploaded_image"],
            uploaded_filename=data["uploaded_filename"],
            fire_count=data["fire_count"],
            smoke_count=data["smoke_count"],
            highest_confidence=data["highest_confidence"],
            detected_objects=data["detected_objects"],
            system_status=data["system_status"]
        )

    # Normal fresh page
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select an image."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Only JPG, JPEG, PNG and WEBP images are allowed."
        )

    filename = secure_filename(file.filename)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save uploaded image
    file.save(file_path)

    # Run Fire + Smoke AI detection
    results = model(file_path)

    fire_count = 0
    smoke_count = 0
    highest_confidence = 0
    detected_objects = []

    # Process detections
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            confidence_percent = confidence * 100

            detected_objects.append({
                "name": class_name,
                "confidence": confidence_percent
            })

            highest_confidence = max(
                highest_confidence,
                confidence_percent
            )

            if class_name.lower() == "fire":
                fire_count += 1

            elif class_name.lower() == "smoke":
                smoke_count += 1

    # Create annotated image with bounding boxes
    annotated_filename = "detected_" + filename

    annotated_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        annotated_filename
    )

    results[0].save(filename=annotated_path)

    # Browser-accessible image URL
    image_url = "/" + annotated_path.replace("\\", "/")

    # Determine system status
    if fire_count > 0:
        system_status = "FIRE DETECTED"

    elif smoke_count > 0:
        system_status = "SMOKE DETECTED"

    else:
        system_status = "SAFE"

    # Store detection results temporarily
    session["detection"] = {
        "uploaded_image": image_url,
        "uploaded_filename": filename,
        "fire_count": fire_count,
        "smoke_count": smoke_count,
        "highest_confidence": highest_confidence,
        "detected_objects": detected_objects,
        "system_status": system_status
    }

    # Redirect to home page
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)