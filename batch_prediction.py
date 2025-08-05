import os
import csv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Configuration — replace with your own values
ENDPOINT = os.getenv("PREDICTION_ENDPOINT", "<your_endpoint>")
PREDICTION_KEY = os.getenv("PREDICTION_KEY", "<your_key>")
PROJECT_ID = "<your_project_id>"
ITERATION_NAME = "<your_published_iteration>"

# Authenticate prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, credentials)

# Define input and output paths
INPUT_FOLDER = "test_images"
OUTPUT_CSV = "batch_predictions.csv"

# Collect all image files in the input folder
image_files = [
    os.path.join(INPUT_FOLDER, fname)
    for fname in os.listdir(INPUT_FOLDER)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
]

with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ["filename", "top_tag", "probability"] + ["tag_"+str(i) for i in range(1,6)]
    writer.writerow(header)

    for image_path in image_files:
        with open(image_path, "rb") as img_data:
            results = predictor.classify_image(PROJECT_ID, ITERATION_NAME, img_data)

        # Sort and pick the top 5 predictions
        sorted_preds = sorted(results.predictions, key=lambda p: p.probability, reverse=True)
        top = sorted_preds[0]
        row = [
            os.path.basename(image_path),
            top.tag_name,
            f"{top.probability:.4f}"
        ] + [
            f"{p.tag_name}:{p.probability:.4f}" for p in sorted_preds[:5]
        ]
        writer.writerow(row)

print(f"✅ Batch prediction complete: {len(image_files)} images → {OUTPUT_CSV}")
