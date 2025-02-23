Music Genre Classification Web App 🎵
A Flask-based web application that classifies music genres from uploaded audio files. The app extracts audio features, visualizes the spectrum, and predicts the genre using a trained machine learning model.

🚀 Features
Upload an audio file and get its genre prediction
Visualize the audio spectrum on the webpage
Machine learning-based classification using Scikit-learn
Music-themed responsive UI
🛠 Tech Stack
Backend: Flask, Python
Machine Learning: Librosa, Scikit-learn
Frontend: HTML, CSS, JavaScript
Dataset: GTZAN
🔧 Installation & Setup
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
Create a virtual environment & install dependencies:
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run the application:
bash
Copy
Edit
python app.py
Open in your browser:
cpp
Copy
Edit
http://127.0.0.1:5000
📂 Project Structure
php
Copy
Edit
📦 music-genre-classification  
 ┣ 📂 static            # CSS, JavaScript, images  
 ┣ 📂 templates         # HTML templates  
 ┣ 📂 models            # Trained ML model  
 ┣ 📂 uploads           # Uploaded audio files  
 ┣ 📜 app.py            # Flask app  
 ┣ 📜 model.py          # ML training script  
 ┣ 📜 requirements.txt  # Dependencies  
 ┗ 📜 README.md         # Project documentation  
📊 Model Training
Features extracted from audio files using Librosa
Trained a classifier using Scikit-learn
Dataset: GTZAN (10 genres, 1,000 samples)
🌎 Deployment
To deploy on Render, Heroku, or AWS, follow their respective Flask deployment guides.

📜 License
This project is licensed under the MIT License.
