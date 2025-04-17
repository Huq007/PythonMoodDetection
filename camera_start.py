from deepface import DeepFace
import cv2
import sys
import time

def analyze_emotion_from_webcam():
    print("Initializing camera...")
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Please check if:")
        print("1. Your camera is connected")
        print("2. You have granted camera permissions in System Preferences > Security & Privacy > Privacy > Camera")
        return
    
    # Wait a bit for the camera to warm up
    print("Camera initialized. Capturing frame in 2 seconds...")
    time.sleep(2)
    
    # Read a frame
    ret, frame = cap.read()
    
    # Release the camera immediately
    cap.release()
    
    if not ret or frame is None:
        print("Error: Could not capture frame from camera")
        return
    
    try:
        print("Analyzing emotion...")
        # Analyze emotion with enforce_detection=False to handle cases where face detection might fail
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Format the results nicely
        print("\nAnalysis Results:")
        print("-----------------")
        print(f"Dominant Emotion: {result[0]['dominant_emotion'].upper()}")
        print("\nEmotion Percentages:")
        for emotion, score in result[0]['emotion'].items():
            print(f"{emotion.capitalize():10}: {score:.2f}%")
        
        # Print face detection confidence
        print(f"\nFace Detection Confidence: {result[0]['face_confidence']:.2f}")
        
        if result[0]['face_confidence'] < 0.5:
            print("\nNote: Low face detection confidence. For better results:")
            print("1. Ensure your face is well-lit")
            print("2. Face the camera directly")
            print("3. Make sure there are no strong shadows on your face")
            
    except Exception as e:
        print("Error during emotion analysis:", str(e))
        print("\nTroubleshooting tips:")
        print("1. Make sure you are facing the camera")
        print("2. Ensure good lighting conditions")
        print("3. Try to position your face clearly in front of the camera")

if __name__ == "__main__":
    analyze_emotion_from_webcam()

# from deepface import DeepFace
# result = DeepFace.analyze(img_path = "sad-face.png", actions = ['emotion'])
# print("========================================")
# print(result)