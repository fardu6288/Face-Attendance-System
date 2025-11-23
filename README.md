Face Attendance System

A lightweight, real-time face recognition attendance system built using OpenCV, NumPy, and Pandas — no GPU, no deep learning, and no heavy models required. Just plug in a webcam and run.

The system detects faces, extracts histogram-based features, compares them with known faces, and logs attendance automatically into a CSV file.

🔥 Features
✔ Real-Time Face Detection

Uses OpenCV Haarcascade to detect faces from a live webcam stream.
(See detector in code) 

main

✔ Offline Face Recognition (No Deep Learning Needed)

Recognition is done using:

Grayscale normalization

Histogram feature extraction

Correlation-based matching


main

✔ Auto Attendance Logging

Each recognized person is logged with:

Name

Date

Time
Saved to attendance.csv, appended across sessions.


main

✔ Snapshot & Reset Controls

While the camera is running:

Q → Quit and save attendance

S → Save snapshot

R → Reset today’s attendance


main

✔ Easy Setup

Just create a folder and drop labeled face images:

known_faces/
    John.jpg
    Sarah.png


Multiple images per person allowed.


main

📁 Project Structure
├── main.py              # Main face attendance system
├── attendance.csv       # Auto-generated attendance log
├── known_faces/         # Add your face images here
├── requirements.txt     # Required Python libraries


requirements.txt includes OpenCV, NumPy, Pandas, etc.


requirements

🛠️ Installation
1. Install dependencies
pip install -r requirements.txt

2. Add face images

Create a known_faces folder next to main.py, then add images named:

Alice.jpg
Bob.png

3. Run the program
python main.py


If faces are loaded, press ENTER to start the camera.

🎮 Key Controls
Key	Action
Q	Quit & save attendance
S	Capture snapshot
R	Reset today’s attendance
🎯 How It Works

Loads Haarcascade face detector

Loads every image in known_faces/

Detects the face → extracts histogram features

Compares features with stored templates

If match > 70% → mark attendance

Writes attendance to CSV

All logic implemented in UniversalFaceAttendance class.


main

📊 Attendance Output (CSV)

Example:

Name	Date	Time
John	2025-11-22	09:13:44
Sarah	2025-11-22	09:14:02
🧰 Tech Stack

OpenCV (face detection)

NumPy (vector features & correlation)

Pandas (CSV storage)

Dependencies defined in requirements.txt.


requirements

📝 Notes

Ensure your webcam is free (Zoom/Teams off).

More images = better recognition accuracy.

Faces must be clear, front-facing, good lighting.
