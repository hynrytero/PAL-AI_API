import sys
import json
import io
import os
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
import warnings

def suppress_stdout_stderr():
    """Suppress stdout and stderr output."""
    import os
    import sys

    # Open file descriptors for stdout and stderr
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    # Backup original stdout and stderr
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    
    # Redirect stdout and stderr to devnull
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    
    return devnull, old_stdout, old_stderr

def restore_stdout_stderr(devnull, old_stdout, old_stderr):
    """Restore original stdout and stderr."""
    # Close the devnull file descriptor
    os.close(devnull)
    
    # Restore original stdout and stderr
    os.dup2(old_stdout, 1)
    os.dup2(old_stderr, 2)
    
    # Close the backup file descriptors
    os.close(old_stdout)
    os.close(old_stderr)

def main():
    try:
        # Suppress all warnings
        warnings.filterwarnings('ignore')

        # Read image data from stdin
        image_data = sys.stdin.buffer.read()
        
        # Validate image data
        if not image_data:
            raise ValueError("No image data received.")
        
        # Load the image
        try:
            image = Image.open(io.BytesIO(image_data))
        except UnidentifiedImageError:
            raise ValueError("Invalid image data provided.")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Load YOLO model
        model_path = os.path.join(os.path.dirname(__file__), 'pal-ai-model', 'best.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Suppress stdout and stderr
        devnull, old_stdout, old_stderr = suppress_stdout_stderr()
        
        try:
            # Load and run model
            model = YOLO(model_path)
            results = model(image)
        finally:
            # Always restore stdout and stderr
            restore_stdout_stderr(devnull, old_stdout, old_stderr)
        
        # Process predictions
        predictions = []
        for result in results:
            for box in result.boxes:
                prediction = {
                    "xmin": float(box.xyxy[0][0].item()),
                    "ymin": float(box.xyxy[0][1].item()),
                    "xmax": float(box.xyxy[0][2].item()),
                    "ymax": float(box.xyxy[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                    "class": int(box.cls[0].item())
                }
                predictions.append(prediction)
        
        # Print only JSON to stdout
        print(json.dumps(predictions))

    except Exception as e:
        # Handle and log errors in JSON format
        error_response = {"error": str(e)}
        print(json.dumps(error_response), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()