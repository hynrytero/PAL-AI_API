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

        # Define class names based on your provided list
        CLASS_NAMES = [
            "Bacterial Leaf Blight",
            "Leaf Blast", 
            "Tungro"
        ]

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
        
        CLASS_DESCRIPTIONS = {
            "Bacterial Leaf Blight": {
                "description": "A devastating bacterial disease that causes lesions on rice leaves, leading to significant yield loss. Symptoms include water-soaked streaks that turn grayish-white, often spreading from leaf tips. High humidity and water-logging contribute to its spread. Proper water management and resistant varieties are key to control.",
                "treatments": [
                    "Use copper-based bactericides to control bacterial spread",
                    "Practice crop rotation to break disease cycle",
                    "Plant resistant rice varieties developed by agricultural research centers",
                    "Maintain proper field drainage to reduce humidity",
                    "Remove and destroy infected plant debris",
                    "Avoid overhead irrigation which can spread bacteria",
                    "Implement balanced nitrogen fertilization",
                    "Use seeds from certified disease-free sources"
                ]
            },
            "Leaf Blast": {
                "description": "A fungal disease caused by Pyricularia oryzae, characterized by diamond-shaped or elliptical lesions on leaves. It can severely impact rice production by reducing leaf area and causing yield losses. Symptoms include gray or brown spots with darker borders. Crop rotation and fungicide treatments can help manage the disease.",
                "treatments": [
                    "Apply fungicides containing tricyclazole or azoxystrobin",
                    "Practice crop rotation with non-host crops",
                    "Use resistant rice cultivars",
                    "Maintain proper spacing between plants for air circulation",
                    "Avoid excessive nitrogen fertilization",
                    "Manage water levels in paddy fields",
                    "Remove and destroy infected plant materials",
                    "Use seed treatments with fungicidal coatings"
                ]
            },
            "Tungro": {
                "description": "A viral disease transmitted by leafhoppers, causing stunted growth and yellowing of rice plants. It's particularly harmful in tropical and subtropical regions. Infected plants show reduced tillering, shorter panicles, and lower grain quality. Vector control and planting resistant varieties are primary management strategies.",
                "treatments": [
                    "Plant virus-resistant rice varieties",
                    "Use yellow sticky traps to control leafhopper populations",
                    "Practice timely planting to avoid peak leafhopper seasons",
                    "Implement biological control with natural leafhopper predators",
                    "Remove alternative host plants near rice fields",
                    "Use insecticides targeted at leafhoppers",
                    "Maintain field sanitation",
                    "Practice crop rotation to disrupt disease cycle"
                ]
            }
        }

        # Modify the prediction processing code
        predictions = []
        for result in results:
            for box in result.boxes:
                class_num = int(box.cls[0].item())
                class_name = CLASS_NAMES[class_num] if 0 <= class_num < len(CLASS_NAMES) else f"Unknown_Class_{class_num}"
                
                prediction = {
                    "xmin": float(box.xyxy[0][0].item()),
                    "ymin": float(box.xyxy[0][1].item()),
                    "xmax": float(box.xyxy[0][2].item()),
                    "ymax": float(box.xyxy[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                    "class": class_num,
                    "class_name": class_name,
                    "description": CLASS_DESCRIPTIONS.get(class_name, {}).get("description", "A rice plant disease affecting crop health and yield."),
                    "treatments": CLASS_DESCRIPTIONS.get(class_name, {}).get("treatments", ["Consult local agricultural experts for specific treatment recommendations"])
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
