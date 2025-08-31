# Facial Recognition Attendance System
# Hinglish comments ke saath - Easy to understand!

import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

class FaceAttendanceSystem:
    def __init__(self):
        """
        System initialize karte hai - sabse pehle yeh chalega
        """
        self.known_faces = []  # Stored faces ki list
        self.known_names = []  # Names ki list
        self.attendance_file = "attendance.csv"  # CSV file ka naam
        self.face_data_file = "face_data.pkl"  # Face data save karne ke liye
        
        # OpenCV ka face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Agar pehle se face data hai to load kar lo
        self.load_face_data()
        
        # CSV file banao agar nahi hai
        self.create_attendance_file()
    
    def create_attendance_file(self):
        """
        Attendance CSV file banata hai agar exist nahi karti
        """
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
            df.to_csv(self.attendance_file, index=False)
            print(f"✅ Attendance file '{self.attendance_file}' create ho gayi!")
    
    def save_face_data(self):
        """
        Face data ko file mein save karta hai
        """
        data = {
            'faces': self.known_faces,
            'names': self.known_names
        }
        with open(self.face_data_file, 'wb') as f:
            pickle.dump(data, f)
        print("✅ Face data save ho gaya!")
    
    def load_face_data(self):
        """
        Saved face data ko load karta hai
        """
        if os.path.exists(self.face_data_file):
            try:
                with open(self.face_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data['faces']
                    self.known_names = data['names']
                print(f"✅ {len(self.known_names)} saved faces load ho gaye!")
            except:
                print("❌ Face data load nahi ho saka")
    
    def capture_and_train_face(self, name):
        """
        Naye person ka face capture karta hai aur train karta hai
        """
        print(f"\n📸 {name} ke liye face capture kar rahe hai...")
        print("Instructions:")
        print("- Camera ke samne properly baithiye")
        print("- Face clear dikhna chahiye")
        print("- Space bar press karke photo le sakte hai")
        print("- ESC press karke exit kar sakte hai")
        
        cap = cv2.VideoCapture(0)  # Camera on karo
        face_samples = []  # Multiple photos store karenge
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera access nahi mil raha!")
                break
            
            # Frame ko flip karo (mirror effect)
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detect karo
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Face ke around rectangle draw karo
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Press SPACE to capture {name}'s face", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Text dikhao
            cv2.putText(frame, f"Capturing for: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Samples captured: {len(face_samples)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Face Capture - Press SPACE to capture, ESC to exit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # Space bar
                if len(faces) > 0:
                    # Pehla face capture karo
                    x, y, w, h = faces[0]
                    face_region = gray[y:y+h, x:x+w]
                    face_region = cv2.resize(face_region, (100, 100))  # Standard size
                    face_samples.append(face_region)
                    print(f"📸 Sample {len(face_samples)} captured!")
                    
                    if len(face_samples) >= 10:  # 10 samples enough hai
                        print("✅ Enough samples captured!")
                        break
                else:
                    print("❌ Koi face nahi mila! Thik se camera ke samne aaiye")
            
            elif key == 27:  # ESC key
                print("❌ Face capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_samples) > 0:
            # Average face nikalo (simple approach)
            avg_face = np.mean(face_samples, axis=0).astype(np.uint8)
            
            self.known_faces.append(avg_face)
            self.known_names.append(name)
            self.save_face_data()
            
            print(f"✅ {name} ka face successfully train ho gaya!")
            print(f"📊 Total registered faces: {len(self.known_names)}")
        else:
            print("❌ Koi face capture nahi hua!")
    
    def recognize_face(self, face_region):
        """
        Face ko recognize karta hai
        Simple template matching use kar rahe hai
        """
        if len(self.known_faces) == 0:
            return "Unknown", 100
        
        face_region = cv2.resize(face_region, (100, 100))
        min_distance = float('inf')
        best_match = "Unknown"
        
        # Har saved face ke saath compare karo
        for i, known_face in enumerate(self.known_faces):
            # Template matching
            result = cv2.matchTemplate(face_region, known_face, cv2.TM_SQDIFF_NORMED)
            distance = result[0][0]
            
            if distance < min_distance:
                min_distance = distance
                best_match = self.known_names[i]
        
        # Threshold set karo - agar distance zyada hai to unknown
        confidence = (1 - min_distance) * 100
        if min_distance > 0.6:  # Adjust kar sakte hai
            return "Unknown", confidence
        
        return best_match, confidence
    
    def mark_attendance(self, name):
        """
        Attendance mark karta hai CSV file mein
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check karo ki aaj already attendance mark hai ya nahi
        try:
            df = pd.read_csv(self.attendance_file)
            today_records = df[(df['Name'] == name) & (df['Date'] == date_str)]
            
            if len(today_records) > 0:
                print(f"⚠️ {name} ki aaj ki attendance already marked hai!")
                return False
        except:
            pass
        
        # New attendance entry
        new_entry = pd.DataFrame({
            'Name': [name],
            'Date': [date_str],
            'Time': [time_str],
            'Status': ['Present']
        })
        
        # CSV mein append karo
        try:
            existing_df = pd.read_csv(self.attendance_file)
            updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
        except:
            updated_df = new_entry
        
        updated_df.to_csv(self.attendance_file, index=False)
        print(f"✅ {name} ki attendance mark ho gayi! Time: {time_str}")
        return True
    
    def start_attendance_system(self):
        """
        Main attendance system start karta hai
        """
        print("\n🎯 Facial Recognition Attendance System Start!")
        print("Instructions:")
        print("- Camera ke samne aaiye")
        print("- Face recognize hone pe automatically attendance mark ho jayegi")
        print("- ESC press karke exit kar sakte hai")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera access nahi mil raha!")
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detect karo
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Face region extract karo
                face_region = gray[y:y+h, x:x+w]
                
                # Face recognize karo
                name, confidence = self.recognize_face(face_region)
                
                # Results display karo
                if name != "Unknown":
                    color = (0, 255, 0)  # Green for known faces
                    label = f"{name} ({confidence:.1f}%)"
                    
                    # High confidence pe attendance mark karo
                    if confidence > 70:  # Threshold adjust kar sakte hai
                        self.mark_attendance(name)
                else:
                    color = (0, 0, 255)  # Red for unknown faces
                    label = f"Unknown ({confidence:.1f}%)"
                
                # Rectangle aur text draw karo
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # System info display karo
            cv2.putText(frame, f"Registered: {len(self.known_names)} people", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC to exit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Attendance System - ESC to exit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Attendance system band ho gaya!")
    
    def show_attendance_report(self):
        """
        Attendance report dikhata hai
        """
        try:
            df = pd.read_csv(self.attendance_file)
            if len(df) == 0:
                print("📋 Abhi tak koi attendance nahi hai!")
                return
            
            print("\n📋 ATTENDANCE REPORT")
            print("=" * 50)
            print(df.to_string(index=False))
            
            # Summary
            print(f"\n📊 SUMMARY:")
            print(f"Total entries: {len(df)}")
            
            # Date-wise summary
            date_summary = df.groupby('Date').size()
            print(f"\n📅 Date-wise attendance:")
            for date, count in date_summary.items():
                print(f"{date}: {count} people")
            
        except FileNotFoundError:
            print("❌ Attendance file nahi mili!")
        except Exception as e:
            print(f"❌ Error reading attendance: {e}")
    
    def list_registered_people(self):
        """
        Registered people ki list dikhata hai
        """
        if len(self.known_names) == 0:
            print("👥 Koi registered person nahi hai!")
            return
        
        print("\n👥 REGISTERED PEOPLE:")
        print("=" * 30)
        for i, name in enumerate(self.known_names, 1):
            print(f"{i}. {name}")
        print(f"\nTotal: {len(self.known_names)} people registered")

def main():
    """
    Main function - Program yahan se start hota hai
    """
    print("🔥 FACIAL RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 50)
    print("Made with ❤️ in Python!")
    print("Hinglish comments ke saath - Easy to understand!")
    
    # System initialize karo
    system = FaceAttendanceSystem()
    
    while True:
        print("\n🏠 MAIN MENU:")
        print("1. Naya person register karo")
        print("2. Attendance system start karo")
        print("3. Attendance report dekho")
        print("4. Registered people dekho")
        print("5. Exit")
        
        try:
            choice = input("\nApna choice enter karo (1-5): ").strip()
            
            if choice == '1':
                name = input("Person ka naam enter karo: ").strip()
                if name:
                    system.capture_and_train_face(name)
                else:
                    print("❌ Valid naam enter karo!")
            
            elif choice == '2':
                if len(system.known_names) == 0:
                    print("⚠️ Pehle koi person register karo!")
                else:
                    system.start_attendance_system()
            
            elif choice == '3':
                system.show_attendance_report()
            
            elif choice == '4':
                system.list_registered_people()
            
            elif choice == '5':
                print("👋 Dhanyawad! System band ho raha hai...")
                break
            
            else:
                print("❌ Invalid choice! 1-5 ke beech mein number enter karo")
        
        except KeyboardInterrupt:
            print("\n\n👋 System interrupted! Bye bye!")
            break
        except Exception as e:
            print(f"❌ Koi error aayi: {e}")

# Program yahan se start hota hai
if __name__ == "__main__":
    main()