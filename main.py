import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

class UniversalFaceAttendance:
    def __init__(self):
        self.known_faces = {}  # {name: [face_templates]}
        self.attendance_log = []
        self.marked_today = set()
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Error: Could not load face cascade!")
        
    def extract_features(self, face_gray):
        """Extract simple features from face image"""
        # Resize to standard size
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Calculate histogram (simple feature)
        hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def compare_faces(self, features1, features2):
        """Compare two face feature vectors"""
        # Calculate correlation
        correlation = np.corrcoef(features1, features2)[0, 1]
        return correlation
    
    def load_known_faces(self, folder_path="known_faces"):
        """Load all face images from the known_faces folder"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created '{folder_path}' folder. Please add face images here.")
            return
        
        print("\n" + "="*60)
        print("Loading known faces...")
        print("="*60)
        
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder_path, filename)
                image = cv2.imread(path)
                
                if image is None:
                    print(f"✗ Could not read: {filename}")
                    continue
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )
                
                if len(faces) > 0:
                    # Get the largest face
                    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Extract features
                    features = self.extract_features(face_roi)
                    
                    name = os.path.splitext(filename)[0]
                    
                    if name not in self.known_faces:
                        self.known_faces[name] = []
                    
                    self.known_faces[name].append(features)
                    print(f"✓ Loaded: {name}")
                else:
                    print(f"✗ No face found in: {filename}")
        
        if self.known_faces:
            print(f"\n✓ Total faces loaded: {len(self.known_faces)}")
            print(f"✓ Ready to start!")
        else:
            print("\n✗ No faces loaded. Please add face images.")
        
        print("="*60)
    
    def recognize_face(self, face_gray):
        """Recognize a face and return name and confidence"""
        features = self.extract_features(face_gray)
        
        best_name = "Unknown"
        best_score = 0
        
        for name, known_features_list in self.known_faces.items():
            # Compare with all stored features for this person
            scores = []
            for known_features in known_features_list:
                score = self.compare_faces(features, known_features)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_name = name
        
        # Threshold for recognition (0.7 = 70% match)
        if best_score > 0.7:
            return best_name, best_score
        else:
            return "Unknown", best_score
    
    def mark_attendance(self, name):
        """Mark attendance for a person"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if already marked today
        if name not in self.marked_today:
            self.marked_today.add(name)
            self.attendance_log.append({
                'Name': name,
                'Date': date_str,
                'Time': time_str
            })
            print(f"✓ Attendance marked for {name} at {time_str}")
            return True
        return False
    
    def save_attendance(self, filename="attendance.csv"):
        """Save attendance to CSV file"""
        if self.attendance_log:
            df = pd.DataFrame(self.attendance_log)
            
            # Append to existing file or create new
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(filename, index=False)
            print(f"\n✓ Attendance saved to {filename}")
            print(f"✓ Total records: {len(df)}")
        else:
            print("\n✗ No attendance to save.")
    
    def run_attendance(self):
        """Main function to run face recognition and attendance"""
        if not self.known_faces:
            print("\n✗ No faces loaded! Please add images to 'known_faces' folder.")
            print("\nSetup Instructions:")
            print("  1. Create a 'known_faces' folder")
            print("  2. Add clear face photos (JPG/PNG)")
            print("  3. Name them like: John.jpg, Sarah.png")
            print("  4. Run the program again")
            return
        
        print("\n" + "="*60)
        print("STARTING FACE ATTENDANCE SYSTEM")
        print("="*60)
        print("Controls:")
        print("  - Press 'Q' to quit and save attendance")
        print("  - Press 'S' to take a snapshot")
        print("  - Press 'R' to reset today's attendance")
        print("="*60 + "\n")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Could not open camera!")
            print("Troubleshooting:")
            print("  - Check if camera is connected")
            print("  - Try: cap = cv2.VideoCapture(1)")
            print("  - Close other apps using the camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Camera opened successfully!")
        
        frame_count = 0
        recognition_cooldown = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to grab frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )
                
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Recognize face
                    name, confidence = self.recognize_face(face_roi)
                    
                    # Set color based on recognition
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green
                        
                        # Mark attendance with cooldown
                        if name not in recognition_cooldown or frame_count - recognition_cooldown[name] > 30:
                            self.mark_attendance(name)
                            recognition_cooldown[name] = frame_count
                    else:
                        color = (0, 0, 255)  # Red
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw name background
                    cv2.rectangle(display_frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                    
                    # Put name
                    cv2.putText(display_frame, name, (x+5, y+h-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Put confidence
                    conf_text = f"{int(confidence * 100)}%"
                    cv2.putText(display_frame, conf_text, (x+5, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display info panel
            cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 0, 0), cv2.FILLED)
            info_text = f"Marked Today: {len(self.marked_today)}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show names
            if self.marked_today:
                names = ", ".join(list(self.marked_today)[:4])
                if len(self.marked_today) > 4:
                    names += "..."
                cv2.putText(display_frame, names, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Attendance System - Press Q to Quit', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Snapshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                self.marked_today.clear()
                recognition_cooldown.clear()
                print("✓ Today's attendance reset!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.save_attendance()
        
        print("\n" + "="*60)
        print("SESSION COMPLETED")
        print("="*60)
        print(f"Total people marked: {len(self.marked_today)}")
        if self.marked_today:
            print(f"Names: {', '.join(self.marked_today)}")
        print("="*60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNIVERSAL FACE ATTENDANCE SYSTEM")
    print("Works with any OpenCV installation!")
    print("="*60)
    
    system = UniversalFaceAttendance()
    system.load_known_faces()
    
    if system.known_faces:
        input("\nPress ENTER to start the camera...")
        system.run_attendance()
    else:
        print("\n⚠️  No faces loaded!")
        print("\nQuick Setup:")
        print("  1. Create 'known_faces' folder in your project")
        print("  2. Add photos: John.jpg, Sarah.png, etc.")
        print("  3. Run: python main.py")