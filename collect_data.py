import cv2
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='data_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define gesture labels and their corresponding keys
GESTURE_MAP = {
    ord('0'): 'fist',
    ord('1'): 'palm',
    ord('2'): 'thumbs_up',
    ord('3'): 'thumbs_down',
    ord('4'): 'peace'
}

# Image setting
IMG_SIZE = 224

def create_dirs():
    """Create directories for each gesture if they don't exist."""
    base_dir = "dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logging.info(f"Created base directory: {base_dir}")

    for key in GESTURE_MAP:
        folder_name = GESTURE_MAP[key]
        path = os.path.join(base_dir, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

def collect_data():
    """Main function to capture video and save images."""
    create_dirs()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        print("Error: Could not open webcam.")
        return

    logging.info("Started data collection session.")
    print("Starting data collection...")
    print("Press 0-4 to save images to corresponding folders.")
    print("Press 'q' to quit.")
    print(f"Mappings: {str({chr(k): v for k, v in GESTURE_MAP.items()})}")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame.")
            print("Error: Failed to capture frame.")
            break

        # Processing for display (keep aspect ratio or just showing raw?)
        # For saving, we need 224x224 grayscale.
        
        # Display the frame
        cv2.imshow('Gesture Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            logging.info("User requested to quit.")
            break
        
        if key in GESTURE_MAP:
            folder_name = GESTURE_MAP[key]
            
            # Process image for saving
            # Resize
            try:
                resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                # Grayscale
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{folder_name}_{timestamp}.jpg"
                save_path = os.path.join("dataset", folder_name, filename)
                
                # Save
                cv2.imwrite(save_path, gray_frame)
                
                log_msg = f"Saved {filename} to {folder_name}"
                logging.info(log_msg)
                print(log_msg)
                
            except Exception as e:
                logging.error(f"Error processing/saving image: {e}")
                print(f"Error: {e}")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Data collection session ended.")

if __name__ == "__main__":
    collect_data()
