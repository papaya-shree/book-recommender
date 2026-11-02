# 📚 Book Recommender System

A machine learning web application built with Streamlit that recommends books using **Popularity-Based Filtering** and **Collaborative Filtering** (Cosine Similarity).

---

## 🚀 Live Demo & Features
- **🔍 Collaborative Recommendation**: Select any of the 700+ popular books to get top matched recommendations based on reading patterns of active readers.
- **🌟 Top 50 Books**: Browse the 50 highest-rated books with at least 250 ratings.
- **⚡ Fast & Lightweight**: Precomputed similarity matrix and optimized data artifacts for instant recommendation.
- **🎨 Responsive UI**: Clean card grid layout with book covers, author details, and match scores.

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Frontend / Deployment**: [Streamlit](https://streamlit.io/)
- **Machine Learning & Data**: Pandas, NumPy, Scikit-learn (Cosine Similarity)
- **Dataset**: [Book-Crossing Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

---

## 📂 Project Structure
```text
Book_Recommender/
│── app.py                         # Streamlit web application
│── Book_Recommender_System.ipynb  # Data analysis & model training notebook
│── popular.pkl                    # Top 50 popular books metadata
│── pt.pkl                         # Pivot table (Book-Title x User-ID)
│── similarity_scores.pkl          # Cosine similarity matrix
│── books.pkl                      # Compact book metadata (titles, authors, images)
│── requirements.txt               # Python package dependencies
│── .gitignore                     # Git ignore rules (ignores heavy CSVs)
└── README.md                      # Documentation
```

---

## 💻 Local Setup & Execution

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
The app will automatically launch in your default web browser at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud

Deploying this app is completely **free** via Streamlit Community Cloud:

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy Book Recommender System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   *(Note: The `.gitignore` ensures the heavy ~107MB CSV files are excluded, so your push will be fast and well below GitHub limits).*

2. **Deploy on Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set the main file path to `app.py`.
   - Click **"Deploy!"**.
   - Streamlit will install the requirements from `requirements.txt` and launch your app live in 1-2 minutes!

