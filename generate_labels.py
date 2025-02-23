import os

IMAGE_FOLDER = "data/captcha_images/"
LABELS_FILE = "data/labels.txt"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Open labels.txt file for writing
with open(LABELS_FILE, "w") as file:
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith(".png"):  # Only process PNG images
            label = os.path.splitext(filename)[0]  # Extract text from filename
            file.write(f"{filename},{label}\n")  # Write to file

print(f"✅ Labels saved in '{LABELS_FILE}'")
