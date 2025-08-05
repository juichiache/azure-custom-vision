import os
import csv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Directory of images
image_folder = "./test_images"

# Authentication
credentials = ApiKeyCredentials(in_headers={"Prediction-key": "YOUR_PREDICTION_KEY"})
predictor = CustomVisionPredictionClient("YOUR_ENDPOINT", credentials)

# Open CSV writer
with open('classification_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'predicted_tag', 'confidence'])

    for filename in os.listdir(image_folder):
        with open(os.path.join(image_folder, filename), 'rb') as img:
            result = predictor.classify_image("project_id", "iteration", img)
            top_pred = max(result.predictions, key=lambda p: p.probability)
            writer.writerow([filename, top_pred.tag_name, top_pred.probability])
