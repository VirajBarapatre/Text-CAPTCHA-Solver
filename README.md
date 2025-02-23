Text CAPTCHA Solver 🔍🤖

This project demonstrates how to solve text-based CAPTCHAs using deep learning with TensorFlow and Python. It involves generating synthetic CAPTCHA images, preprocessing them, training a neural network, and testing the model to recognize and solve CAPTCHAs automatically.

Features ✨

CAPTCHA Generation: Generate synthetic CAPTCHA images with random characters.

Dataset Preparation: Preprocess images and prepare the dataset for training.

Model Training: Train a Convolutional Neural Network (CNN) to solve CAPTCHAs.

Evaluation & Testing: Evaluate the model's accuracy and performance in solving CAPTCHAs.



---

Prerequisites 📦

Before you begin, ensure that you have Python 3.x installed and the following libraries:

TensorFlow

OpenCV

NumPy

Matplotlib

Pillow


To install the required dependencies, run:

pip install -r requirements.txt


---

Setup & Usage 🛠️

1. Generate CAPTCHA Dataset

To generate a dataset of CAPTCHA images, run the generate_captcha.py script. You can specify the number of images and their size.

python scripts/generate_captcha.py --num_images 1000 --image_size (200, 50)

This will generate 1000 CAPTCHA images in the data/generated_captchas/ directory.

2. Preprocess the Data

Once the images are generated, preprocess them (resize, grayscale, label encoding) for training. Run the following command:

python scripts/preprocess_data.py

This will save the preprocessed data for training in the proper format.

3. Train the Model

Train the neural network model using the preprocessed data. The model will be saved in the models/ directory.

python scripts/train_model.py --epochs 50 --batch_size 32

You can adjust the number of epochs and batch size according to your preference.

4. Evaluate the Model

After training, evaluate the model's performance on the dataset using the following command:

python scripts/evaluate_model.py

This will output the model’s accuracy and performance.

5. Solve a CAPTCHA

You can use the trained model to solve new CAPTCHA images. Use the solve_captcha.py script to predict the text in an image.

python scripts/solve_captcha.py --image_path 'data/generated_captchas/sample_captcha.png'

The model will output the predicted CAPTCHA text.


---

Model Architecture 🧠

The model uses a Convolutional Neural Network (CNN), which is well-suited for image recognition tasks. It consists of:

Convolutional layers for feature extraction.

Dropout layers to prevent overfitting.

Dense layers for classification of CAPTCHA characters.



---

Example Usage 🚀

1. Generate a CAPTCHA image:



python scripts/generate_captcha.py --num_images 1 --image_size (200, 50)

2. Solve the generated CAPTCHA:



python scripts/solve_captcha.py --image_path 'data/generated_captchas/generated_captcha.png'


---

Results 📊

The model is trained and tested on synthetic CAPTCHA images. You can evaluate its performance based on accuracy and its ability to solve unseen CAPTCHA images.


---

Contributing 🤝

Feel free to contribute by forking the repository, fixing bugs, or adding new features. Contributions are welcome!

1. Fork the repository


2. Create a new branch (git checkout -b feature-name)


3. Make your changes and commit them (git commit -am 'Add new feature')


4. Push to the branch (git push origin feature-name)


5. Open a pull request




---

License 📜

This project is licensed under the MIT License - see the LICENSE file for details.


---

Let me know if you want to add anything else or make changes!

