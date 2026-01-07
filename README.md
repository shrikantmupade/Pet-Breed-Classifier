# Pet Breed Classifier (End-to-End Deep Learning Project)

A full-stack computer vision application deployed on the cloud. The system classifies pet breeds from user-uploaded images, utilizing a deep learning model optimized for low-resource inference.

**Live Demo:** [https://pet-breed-classifier.onrender.com](https://pet-breed-classifier.onrender.com)
*(Note: Hosted on Render Free Tier. Spin-down delays of ~45s may occur on the first request.)*

---

## Architecture & Model Pipeline

### 1.Data Pipeline
* **Source:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* **Size:** 7,390 images across 37 classes (Breeds).
* **Split:** 80% Training (5,912), 10% Validation (739), 10% Test (739).
* **Preprocessing:**
    * Images resized to 224x224 (MobileNetV2 standard).
    * Pixel normalization (scaled to [-1,1]).
    * **Data Augmentation:** Random rotation (20°),zoom (20%),shear and horizontal flips were applied during training to improve generalization.

### 2.Model Architecture
I utilized **Transfer Learning** to maximize accuracy with limited training data.
* **Base Model:** **MobileNetV2** (Pre-trained on ImageNet).
    * *Why:* Because it is Lightweight (3.5M params) and fast , making it ideal for the free-tier cloud deployment.
    * *State:* Frozen (Non-trainable) to act as a feature extractor.
* **Custom Head:**
    * `GlobalAveragePooling2D`
    * `Dense` (1024 units,ReLU activation)
    * `Dropout` (0.2)–to prevent overfitting.
    * `Output Layer` (37 units, Softmax activation).
* **Optimizer:** Adam (Learning rate:0.0001).
* **Loss Function:** Categorical Crossentropy.

### 3.Backend & Deployment
The model is served via a **Flask** API. 
* **Challenge:** The cloud environment (Render Free Tier) has a hard limit of 512MB RAM. TensorFlow consumes ~300MB just on import.
* **Solution:** I have Implemented **Lazy Loading** and aggressive Garbage Collection (`gc.collect`). The model is only loaded into memory when a prediction is requested and memory is cleared immediately after inference to prevent OOM (Out of Memory) crashes.

---

## Performance & Limitations

### Training Results
* **Final Test Accuracy:** **91.0%**
* **Validation Accuracy:** 91.07%
* **Training Time:** Converged in 12 epochs(Early Stopping triggered).

### Model Behavior (Strengths & Weaknesses)
After Analysing F1-Scores and Confusion Matrix , the model shows distinct behavioral patterns:

* **Highly Accurate (95-100%):**
    * **Distinctive Features:** The model achieves near-perfect scores on breeds with unique fur textures or shapes , such as the **Samoyed (100%)** , **Pug (97%)** , **Saint Bernard (100%)** and **Yorkshire Terrier (100%)**.
    
* **Common Confusion (The "Bully" Cluster):**
    * **Pit Bull vs. Bulldog vs. Staffie:** The model struggles to differentiate between the **American Pit Bull Terrier** , **American Bulldog** and **Staffordshire Bull Terrier**.
    * *Data Evidence:* Precision for American Bulldog drops to **69%** , and Recall for Staffordshire Bull Terrier is only **68%**. These breeds share very similar facial structures and muscular builds, leading to cross-classification errors.

---

##  Tech Stack

* **Core:** Python 3.10
* **ML Framework:** TensorFlow 2.15 (Keras API), NumPy.
* **Web Framework:** Flask, Gunicorn.
* **Image Processing:** Pillow, Werkzeug.
* **Infrastructure:** Docker (Containerization), Render (Cloud PaaS).

---

##  Project Structure

```text
Pet-Breed-Classifier/
├── app.py                  # Main application logic (Flask + Inference)
├── aaidos.ipynb            # Jupyter notebook used for training
├── requirements.txt        # Pinned dependencies (CPU-optimized for cloud)
├── static/
│   ├── uploads/            # Temporary storage for inference
│   └── images/             # UI assets
├── templates/
│   └── index.html          # Frontend interface
└── animal_breed_model.h5   # Trained weights
```

---

##  Local Installation

If you want to run this app on your own computer:

1. **Clone the repository**
    ```bash
    git clone https://github.com/shrikantmupade/Pet-Breed-Classifier.git
    cd Pet-Breed-Classifier
    ```

2. **Install dependencies**
    ```bash
    # It is recommended to use a virtual environment
    pip install -r requirements.txt
    ```

3. **Run the application**
    ```bash
    python app.py
    ```

4. **Open in Browser**
    Go to `http://127.0.0.1:5000` to see the app running locally.

---

## License
Distributed under the **MIT License**. This means you can use, modify, and distribute this code freely, provided you include the original license file. See `LICENSE` for more information.

---

## Author

**Shrikant Mupade**

* **LinkedIn:** [Connect with me](www.linkedin.com/in/shrikant-mupade-504b17256)
* **GitHub:** [View my work](https://github.com/shrikantmupade)

---
