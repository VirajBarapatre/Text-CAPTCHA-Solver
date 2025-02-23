import os
import random
from captcha.image import ImageCaptcha

# Parameters
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Allowed characters
CAPTCHA_LENGTH = 6  # Length of each CAPTCHA
NUM_IMAGES = 10000  # Number of images to generate
IMAGE_FOLDER = "data/captcha_images/"  # Output folder

# Ensure the directory exists
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Generate images
for i in range(NUM_IMAGES):
    text = "".join(random.choices(CHARACTERS, k=CAPTCHA_LENGTH))  # Generate random text
    image = ImageCaptcha(width=200, height=80)  # Create CAPTCHA generator
    image_path = os.path.join(IMAGE_FOLDER, f"{text}.png")  # Save path
    image.write(text, image_path)  # Save image

print(f"✅ Generated {NUM_IMAGES} CAPTCHA images in '{IMAGE_FOLDER}'")
