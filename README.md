# NLP-Final-Project
ITM-454: NLP Course Final Project

## Project Overview: Harmful Hate Speech Detection
This project implements a Hate, Offensive, and Neutral/Neither text classifier using a dataset from HuggingFace (https://huggingface.co/datasets/tdavidson/hate_speech_offensive), TF-IDF vectorization, and a Logistic Regression model. It includes full preprocessing (emoji handling, stopwords, POS tagging, lemmatization, etc.), dataset balancing, model training, evaluation, and a simple Streamlit interface for demonstration purpose.

---

## Project Structure
project/
│── archive/
    │── data/
        │── labeled_data.csv
        │── labeled_data.p
        │── readme.md
    │── LICENSE
│── hate_speech_detection_LR.ipynb
│── hate_speech_model.pkl
│── interface.py 
│── README.md
│── requirements.txt         
│── vectorizer.pkl   

---

## Team Members
* chesda-ly (Ly Sopheakchesda)
* bongchanbormey (Bong Chan Bormey)
* Julie-lou (Lou Julie) 

---

## Installation and Usage  

### **1. Installation**
1. Clone the repository:  
   ```bash
   git clone https://github.com/bongchanbormey/NLP-Final-Project.git 
   ```
2. Create and activate a virtual environment (recommended)
- Windows:
    python -m venv venv
    venv\Scripts\activate
- MacOS:
    python3 -m venv venv
    source venv/bin/activate
3. Install the required packages:
pip install -r requirements.txt

--- 

### **2. Usage**  
How to Run the Notebook:
- Open Jupyter Notebook (in IDE like VSCode) or Google Colab (make sure that you upload the dataset)
- Load hate_speech_detection_LR.ipynb
- Run all cells in order until the last cell
- To test out the interface, enter 'streamlit run interface.py' in the terminal. Streamlit will pop up in your browser, then enter a sentence, click on 'analyze'. A prediction and confidence scores will show as the outputs.

---

## Acknowledgements
As a team, we'd like to extend our appreciation to:
- Professor Monyrath for her teaching and guidance throughout this course and project.
- Everyone on the team for their contributions to this project.