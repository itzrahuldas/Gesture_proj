import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants
DATA_DIR = 'data'
IMG_SIZE = 128
CLASSES = ['fist', 'palm', 'thumbs_up', 'none']

def create_dirs():
    """Create dataset directories."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for class_name in CLASSES:
        path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(path):
            os.makedirs(path)

def get_counts():
    """Get current image counts for each class."""
    counts = {}
    for class_name in CLASSES:
        path = os.path.join(DATA_DIR, class_name)
        # Count files
        if os.path.exists(path):
            counts[class_name] = len([f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')])
        else:
            counts[class_name] = 0
    return counts

def main():
    create_dirs()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting data collection...")
    print("Press 'f' for Fist, 'p' for Palm, 't' for Thumbs Up, 'n' for None.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Copy frame for display
        display_frame = frame.copy()
        
        # Get counts for overlay
        counts = get_counts()
        
        # Overlay counts
        y_offset = 30
        for i, class_name in enumerate(CLASSES):
            text = f"{class_name}: {counts[class_name]}"
            cv2.putText(display_frame, text, (10, y_offset + (i * 30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Data Collection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        save_class = None
        if key == ord('f'):
            save_class = 'fist'
        elif key == ord('p'):
            save_class = 'palm'
        elif key == ord('t'):
            save_class = 'thumbs_up'
        elif key == ord('n'):
            save_class = 'none'
        elif key == ord('q'):
            break
            
        if save_class:
            # Process image: Resize -> Grayscale
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Save
            count = counts[save_class]
            filename = f"{os.path.join(DATA_DIR, save_class, str(count))}.jpg"
            cv2.imwrite(filename, gray)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
