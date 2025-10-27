# Crop Disease Detection 🌱

A machine learning-powered web application for detecting diseases in crop images using deep learning models.

## 🚀 Features

- **Real-time Disease Detection**: Upload crop images and get instant disease predictions
- **Multiple Model Support**: Utilizes two trained models for accurate detection
- **User-friendly Interface**: Built with Streamlit for easy interaction
- **High Accuracy**: Deep learning models trained on extensive crop disease datasets
- **Fast Processing**: Optimized for quick image analysis and results

## 🛠️ Technology Stack

- **Frontend**: Html, CSS, JS
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Transfer-learning-fine-tune
- **Deployment**: Render

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gautam7906/Crop-disease-detection.git
   cd Crop-disease-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained models**
   - Model 1: [Download here](https://drive.google.com/file/d/1Bgktu20eqkmOPkilaXOuT72K8m3acX-W/view?usp=sharing)
   - Model 2: [Download here](https://drive.google.com/file/d/1eq8kAgY719ZdwOCsVCq37GDJd-a169Ks/view?usp=sharing)
   
   Place the downloaded models in the `models/` directory.

4. **Run the application**
   ```bash
    python app.py
   ```

## 📁 Project Structure

```
Crop-disease-detection/
├── app.py                 # Flask application
├── models/               # Trained ML models
│   ├── model1.keras
│   └── model2.keras
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── assets/              # Static files (images, etc.)
└── utils/               # Helper functions
```

## 🎯 Usage

1. **Start the application**: Run `python app.py`
2. **Upload an image**: Use the file uploader to select a crop image
3. **Get predictions**: The model will analyze the image and provide disease detection results
4. **View results**: See the predicted disease class and confidence score

## 🤖 Models

This project uses two trained deep learning models:

- **Model 1**: Specialized for common crop diseases
- **Model 2**: Enhanced model for broader disease classification

Both models are trained on comprehensive datasets and provide high accuracy in disease detection.

## 🌐 Live Demo

The application is deployed on Render. [View Live Demo](https://crop-disease-detection-2auo.onrender.com) 

## 📊 Supported Crops & Diseases

- Tomato diseases (Early Blight, Late Blight, Leaf Mold, etc.)
- Potato diseases (Early Blight, Late Blight, etc.)
- Corn diseases (Common Rust, Gray Leaf Spot, etc.)
- And many more...

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIET License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Gautam**
- GitHub: [@gautam7906](https://github.com/gautam7906)

## 🙏 Acknowledgments

- Thanks to the agricultural research community for providing datasets
- Special thanks to the open-source community for the tools and libraries
- Inspired by the need to help farmers identify crop diseases early

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the [Issues](https://github.com/gautam7906/Crop-disease-detection/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about the problem

---

⭐ **Star this repository if you found it helpful!** ⭐
