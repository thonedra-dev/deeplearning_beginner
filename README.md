# 🧠 Deep Learning Beginner

<div align="center">

![Deep Learning Banner](https://img.shields.io/badge/Deep%20Learning-Beginner's%20Guide-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

### *Your Journey into Deep Learning Starts Here*

**A comprehensive collection of hands-on notebooks designed to guide beginners through fundamental deep learning concepts and practical implementations.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

[Overview](#-overview) • [Learning Path](#-learning-path) • [Quick Start](#-quick-start) • [Models](#-model-implementations) • [Prerequisites](#-prerequisites)

</div>

---

## 📋 Overview

Welcome to **Deep Learning Beginner** - a carefully curated learning repository that bridges the gap between theory and practice. Whether you're starting from scratch or looking to solidify your understanding, this collection provides progressive, hands-on experience with essential deep learning concepts.

### 🎯 What You'll Learn

- **NumPy Fundamentals**: Master array operations, broadcasting, and mathematical functions essential for deep learning
- **Neural Network Basics**: Understand the building blocks of neural networks through practical examples
- **Image Classification**: Build and train real models on industry-standard datasets
- **TensorFlow/Keras**: Get comfortable with the leading deep learning framework
- **Model Training Pipeline**: Learn data preprocessing, model building, training, and evaluation

---

## 🗺️ Learning Path

This repository is structured as a progressive learning journey. Follow the sequence below for optimal understanding:

### Phase 1: Foundation (NumPy Mastery)
Build your computational foundation with NumPy, the backbone of all deep learning frameworks.

| Notebook | Focus Area | Key Concepts |
|----------|-----------|--------------|
| `day_one_one.py` | Array Basics | Array creation, indexing, slicing |
| `day_one_two.py` | Array Operations | Element-wise operations, broadcasting |
| `day_one_three.py` | Mathematical Functions | Linear algebra, statistics, aggregations |
| `day_one_four.py` | Advanced Techniques | Reshaping, stacking, advanced indexing |
| `day_one_five.py` | Broadcasting & Constraints | Broadcasting rules, memory optimization |

### Phase 2: Deep Learning Foundations
Understand core concepts that power neural networks.

| Notebook | Focus Area | Key Concepts |
|----------|-----------|--------------|
| `day_two_one.py` | Array Manipulation | Multi-dimensional arrays, transpose operations |
| `day_two_two.py` | Matrix Operations | Dot products, matrix multiplication |
| `day_two_three.py` | Array Functions | Swapping axes, complex transformations |
| `day_two_four.py` | Advanced Swapping | 2D/3D array manipulation techniques |

### Phase 3: Advanced Concepts
Master the mathematical operations behind neural networks.

| Notebook | Focus Area | Key Concepts |
|----------|-----------|--------------|
| `day_three_one.py` | 3D Array Processing | Working with image-like data structures |
| `day_three_two.py` | Row-wise & Column-wise Ops | Aggregations across different axes |
| `day_three_three.py` | Max Functions | Finding maxima across dimensions |

### Phase 4: Practical Implementation
Apply your knowledge by building real image classification models.

| Model | Dataset | Task | Complexity |
|-------|---------|------|------------|
| **MNIST** | Handwritten Digits | 10-class classification | ⭐ Beginner |
| **Fashion-MNIST** | Clothing Items | 10-class classification | ⭐⭐ Intermediate |
| **CIFAR-10** | Natural Images | 10-class classification | ⭐⭐⭐ Advanced |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Beginners)

1️⃣ **Clone or Download this Repository**
```bash
git clone https://github.com/thonedra-dev/deeplearning_beginner.git
```

2️⃣ **Upload to Google Drive** (Optional)
- Upload the repository folder to your Google Drive for easy access

3️⃣ **Open in Google Colab**
- Navigate to [Google Colab](https://colab.research.google.com/)
- Select `File` → `Upload notebook` or `Open notebook` → `GitHub`
- Upload individual `.py` or `.ipynb` files

4️⃣ **Run and Learn!**
- Execute cells sequentially
- Experiment with parameters
- Observe outputs and understand behavior

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/thonedra-dev/deeplearning_beginner.git
cd deeplearning_beginner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

---

## 📚 Model Implementations

### 🔢 MNIST - Handwritten Digit Recognition
**File:** `mnist_model.py`

The classic starting point for image classification. Learn to build a neural network that recognizes handwritten digits (0-9).

**What You'll Build:**
- Simple feedforward neural network
- Convolutional Neural Network (CNN)
- Model evaluation and visualization

**Dataset:** 60,000 training images + 10,000 test images (28×28 grayscale)

---

### 👕 Fashion-MNIST - Clothing Classification
**File:** `fashion_mnist.py`

Step up the challenge with more complex grayscale images of clothing items.

**What You'll Build:**
- Enhanced CNN architecture
- Data augmentation techniques
- Performance optimization strategies

**Dataset:** 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

---

### 🖼️ CIFAR-10 - Natural Image Classification
**File:** `ciphar10_model.py`

Master color image classification with real-world complexity.

**What You'll Build:**
- Deep CNN with multiple layers
- Batch normalization and dropout
- Advanced training techniques

**Dataset:** 60,000 color images (32×32 RGB) across 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)

---

## 🛠️ Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of functions, loops, and data structures
- High school mathematics (algebra, basic calculus helpful but not required)

### Software Requirements
- Python 3.9 or higher
- Google Colab account (free) **OR**
- Jupyter Notebook installed locally

### Key Libraries
```txt
numpy>=1.21.0
tensorflow>=2.10.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

> 💡 **Pro Tip:** Google Colab comes with all these libraries pre-installed!

---

## 📁 Repository Structure

```
deeplearning_beginner/
├── day_one_one.py          # NumPy basics: arrays and indexing
├── day_one_two.py          # Broadcasting fundamentals
├── day_one_three.py        # Mathematical operations
├── day_one_four.py         # Advanced array techniques
├── day_one_five.py         # Broadcasting constraints
├── day_two_one.py          # Array classification concepts
├── day_two_two.py          # Matrix operations
├── day_two_three.py        # Swapping and transposing
├── day_two_four.py         # Complex transformations
├── day_three_one.py        # 3D array understanding
├── day_three_two.py        # Axis-wise operations
├── day_three_three.py      # Max functions across arrays
├── mnist_model.py          # MNIST digit classification
├── fashion_mnist.py        # Fashion-MNIST implementation
├── ciphar10_model.py       # CIFAR-10 color images
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🎓 Learning Tips

### For Complete Beginners
1. **Don't rush** - Spend time understanding each NumPy operation
2. **Experiment freely** - Modify code and observe results
3. **Visualize often** - Use matplotlib to see what's happening
4. **Start simple** - Begin with MNIST before moving to CIFAR-10

### Best Practices
- ✅ Run notebooks sequentially from Day 1
- ✅ Complete all NumPy exercises before models
- ✅ Take notes on concepts you find challenging
- ✅ Try modifying model architectures
- ✅ Compare results across different approaches

### Common Pitfalls to Avoid
- ❌ Skipping NumPy fundamentals
- ❌ Not understanding array shapes
- ❌ Rushing to complex models
- ❌ Ignoring error messages

---

## 🤝 Contributing

Found a bug or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📫 Contact & Support

<div align="center">

### **Thonedra**

[![Email](https://img.shields.io/badge/Email-thonedra.dev%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thonedra.dev@gmail.com)

*Passionate about making deep learning accessible to everyone*

</div>

### Need Help?

- 📧 Email me with questions: thonedra.dev@gmail.com
- 🐛 Report issues in the [Issues](https://github.com/thonedra-dev/deeplearning_beginner/issues) section
- 💬 Start a discussion in [Discussions](https://github.com/thonedra-dev/deeplearning_beginner/discussions)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2025 Thonedra
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🌟 Acknowledgments

- **TensorFlow Team** - For the incredible deep learning framework
- **NumPy Community** - For the foundation of scientific computing
- **Google Colab** - For providing free GPU access
- **Deep Learning Community** - For continuous inspiration and support

---

## 📊 Progress Tracking

Use this checklist to track your learning journey:

- [ ] Completed all Day 1 notebooks (NumPy basics)
- [ ] Completed all Day 2 notebooks (Array operations)
- [ ] Completed all Day 3 notebooks (Advanced concepts)
- [ ] Built and trained MNIST model
- [ ] Built and trained Fashion-MNIST model
- [ ] Built and trained CIFAR-10 model
- [ ] Achieved >95% accuracy on MNIST
- [ ] Achieved >85% accuracy on Fashion-MNIST
- [ ] Achieved >70% accuracy on CIFAR-10

---

<div align="center">

### 🎯 Ready to Start Your Deep Learning Journey? 🎯

**Clone this repo and begin with `day_one_one.py`!**

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)

**Built with passion for aspiring AI engineers | © 2025 Thonedra**

</div>