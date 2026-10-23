
# Face Attendance System

A lightweight, real-time face recognition attendance system built with Python, OpenCV, NumPy, and Pandas. The system uses a webcam to detect and recognize known faces and records attendance in a CSV file without requiring a GPU or deep-learning models.

## Features

- **Real-Time Face Detection:** Detects faces from a live webcam feed using OpenCV Haar cascades.
- **Histogram-Based Recognition:** Extracts grayscale histogram features and compares them with stored face templates using correlation-based matching.
- **Automated Attendance Logging:** Records recognized individuals with their names, dates, and timestamps.
- **CSV Storage:** Saves attendance records in `attendance.csv` across sessions.
- **Snapshot Capture:** Press `S` to save a camera snapshot.
- **Attendance Reset:** Press `R` to reset the current day's attendance.
- **Simple Setup:** Add labeled face images to the `known_faces/` directory.
- **CPU-Friendly Processing:** Designed to run without a dedicated GPU or deep-learning model.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| OpenCV | Webcam access and face detection |
| NumPy | Histogram features and numerical operations |
| Pandas | Attendance data management |
| Haar cascades | Face detection |
| CSV | Attendance storage |

## Project Structure

```text
face-attendance-system/
├── main.py
├── requirements.txt
├── known_faces/
│   ├── Alice.jpg
│   └── Bob.png
└── attendance.csv
```

The `attendance.csv` file is generated automatically when attendance is recorded, if implemented that way in the current code.

## Getting Started

### Prerequisites

- Python 3.x
- A webcam
- The libraries listed in `requirements.txt`

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Known Faces

Create a `known_faces/` directory alongside `main.py`.

Add clear face images using each person's name as the filename:

```text
known_faces/
├── Alice.jpg
├── Bob.png
└── Sarah.jpg
```

Use additional images for a person only if the implementation supports multiple images per individual.

### 4. Run the Application

```bash
python main.py
```

If prompted, press Enter to start webcam attendance.

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit and save attendance |
| `S` | Capture a snapshot |
| `R` | Reset today's attendance |

## How It Works

1. Loads the Haar cascade face detector.
2. Reads the images stored in `known_faces/`.
3. Captures frames from the webcam.
4. Detects faces in each frame.
5. Converts detected face regions to grayscale and extracts histogram-based features.
6. Compares the extracted features with stored templates using correlation-based matching.
7. Applies the configured matching threshold to identify potential matches.
8. Records recognized individuals in the attendance CSV file.

The system uses histogram similarity rather than a deep-learning face-recognition model. Its recognition performance may vary with lighting, face orientation, image quality, and similarity between individuals.

## Attendance Output

Example `attendance.csv`:

```csv
Name,Date,Time
Alice,2025-11-22,09:13:44
Bob,2025-11-22,09:14:02
```

Actual dates and timestamps are generated when attendance is recorded.

## Limitations

- Histogram-based matching is less robust than modern face-recognition embeddings.
- Lighting changes and face orientation can affect matching.
- Similar-looking faces may be confused.
- A similarity threshold is not equivalent to a measured recognition accuracy.
- The system should not be treated as a security-grade identity verification solution.

## Privacy Considerations

Face images and attendance records contain personal information. Obtain appropriate consent before collecting or processing them, restrict access to stored images and logs, and follow applicable privacy requirements.

## Future Improvements

- Evaluate recognition performance on a labeled test dataset.
- Add configurable recognition thresholds.
- Improve duplicate-attendance handling.
- Add an attendance summary dashboard.
- Explore more robust face-recognition methods.

## Author

**Muhammad Fardan Naheem**

GitHub: [fardu6288](https://github.com/fardu6288)
