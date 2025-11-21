# 🧠 Gpt2Vizualizer

**Gpt2Vizualizer** is an interactive tool for visualizing the hidden states and token representations of GPT-2 and compatible causal language models. Leveraging Gradio, Plotly, and PCA, it provides intuitive insights into how transformer models process text.

---

## 🚀 Introduction

Understanding how large language models interpret and transform input data can be challenging. **Gpt2Vizualizer** bridges this gap by offering a visual exploration of token embeddings and hidden states. Whether you're a researcher, data scientist, or enthusiast, this project helps you peek under the hood of GPT-2 models in real time.

---

## ✨ Features

- **Interactive Gradio Interface**  
  Simple web UI to input text and select models.

- **Dynamic Visualization**  
  Explore token and hidden state representations with plotly-powered PCA projections.

- **Supports Multiple Models**  
  Easily switch between GPT-2 variants (default: `distilgpt2`).

- **Token-wise Analysis**  
  View how each token is processed through the model’s layers.

- **Device Selection**  
  Automatically detects and utilizes available GPU resources for faster computation.

---

## 🛠️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/pranesh-2005/Gpt2Vizualizer.git
   cd Gpt2Vizualizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   **Or, manually install required packages:**
   ```bash
   pip install gradio torch numpy plotly pandas scikit-learn transformers
   ```

---

## 📈 Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open the Gradio UI**
   - Follow the link shown in your terminal (usually http://localhost:7860).
   - Type your text, select a model, and explore the visualizations.

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> **Gpt2Vizualizer** – Making transformer models transparent, one token at a time!

## License
This project is licensed under the **MIT** License.

---
🔗 GitHub Repo: https://github.com/Pranesh-2005/Gpt2Vizualizer
