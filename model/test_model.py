from ultralytics import YOLO

# Load our Fire + Smoke model
model = YOLO("model/best.pt")

# Test the model on an uploaded image
results = model("static/uploads/Screenshot_2026-09-08_223334.png", save=True)

print("AI TEST COMPLETE!")

for result in results:
    print("Detected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(f"{class_name}: {confidence * 100:.2f}%")