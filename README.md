# Adversarial-Image-Classifier

Adversarial Image Classifier Stress Test
An interactive web application built with PyTorch and Streamlit to test the robustness of a pre-trained computer vision model against common image distortions, or "aggressors."

[➡️ Live Demo Link Here] (https://adversarial-image-classifier.streamlit.app/)

🎯 Project Goal
This project demonstrates the core responsibilities of a Machine Learning Algorithm Validation Engineer. Instead of just building a model, this tool is designed to find its weaknesses and quantify its performance under non-ideal conditions, directly simulating the process of ensuring an algorithm is ready for a real-world production environment.

It is a direct response to the need for engineers who can design and execute "live test procedures, aggressor searches, user studies, and annotation pipelines to improve and influence ML algorithm performance and design."

✨ Why This Project?
This application was built to showcase key qualifications for the Algorithm Validation role at Apple:

Directly Implements "Aggressor Searches": The job description emphasizes the need to design "aggressor searches." This tool's core feature is a set of interactive sliders that apply aggressors (brightness, noise, rotation) to an input image, allowing for immediate analysis of the model's response.

Focuses on End-to-End System Performance: The tool visualizes the entire pipeline from user input to final prediction. This demonstrates an understanding of how to "evaluate and represent the true customer experience" by showing how real-world image imperfections can degrade model performance.


Combines Full-Stack and ML Skills: My background includes developing full-stack applications and working with AI/ML models in Python. This project merges those skills by creating a user-facing dashboard to validate a complex backend model, demonstrating the ability to build practical tools for ML evaluation.

Hands-On Python and ML Framework Experience: The project is built entirely in Python, using PyTorch for the machine learning model and Streamlit for the web framework, showcasing strong programming skills as required.

🚀 Features
Image Uploader: Upload any JPG, JPEG, or PNG image.

Real-Time Aggressor Controls: Use sliders to instantly apply:

Brightness adjustments

Rotational changes

Gaussian noise

Side-by-Side Comparison: View the original and transformed images next to each other.

Dynamic Prediction Analysis: The model's prediction and confidence score update in real-time for both images, with clear visual cues (red for changed predictions, orange for dropped confidence) to highlight model failures.

🛠️ Tech Stack
Language: Python

Machine Learning: PyTorch, Torchvision

Web Framework: Streamlit

Libraries: Pillow, NumPy, Requests

⚙️ Setup and Installation
To run this project locally, follow these steps:

Clone the repository:

Bash

git clone <your-repo-url>
cd <your-repo-name>
Create and activate a Python virtual environment:

Bash

# Create the environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
Install the required dependencies:

Bash

pip install -r requirements.txt
▶️ How to Run
With the virtual environment active, run the following command in your terminal:

Bash

streamlit run app.py
Your web browser will open with the application running.

🖼️ Screenshot
<img width="1666" height="1272" alt="image" src="https://github.com/user-attachments/assets/0dcdbfac-9b48-4b0e-9738-a3236396d8b2" />
