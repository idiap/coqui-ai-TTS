# 🐸🎤 XTTS Fine-Tuning 🖥️

Welcome to the XTTS model fine-tuning repository! This project allows you to fine-tune XTTS (Cross-lingual Text-To-Speech) models.

---

## 💻 About

This project is focusing on the fine-tuning of XTTS models for TTS applications in a gradio web interface. The repository includes model compression techniques to optimize model performance. The main file you will run is `xtts_demo.py.py`.


---

## 🚀 Installation

Follow these steps to set up the project on your machine:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   ### If runing on apple silicon.
   
   The installation requires the `no-dependencies` option since it's built from a `pip freeze`.
   ```bash
   pip install --no-dependencies -r apple_silicon_requirements.txt
   ```

---

## 🐳 Docker usage 

- Docker Usage on x86 Nvidia GPU (You need 12 gb Vram at minimum)
```docker
docker run --gpus all -it -v ${PWD}/training:/tmp/xtts_ft/ athomasson2/fine_tune_xtts:v5
```

- Docker Usage for cpu only docker ( Will crash unless you go into your docker settings and bump your docker resource allocation limit all the way up. )
```docker
docker run -it -v ${PWD}/training:/tmp/xtts_ft/run/training athomasson2/fine_tune_xtts:v4_cpu
```

- Docker Usage on Apple Silicon computers ( Make sure to bump your docker resource allocation limit all the way up. )
```docker
docker run -it -v ${PWD}/training:/tmp/xtts_ft/ athomasson2/fine_tune_xtts:M1
```

- Taken from [This Dockerhub Page](https://hub.docker.com/r/athomasson2/fine_tune_xtts)

## 🛠️ Usage

To fine-tune and run the XTTS model, use the provided demo script.

### **Run the Fine-Tuning Script**
```bash
python xtts_demo.py
```

### **Available Arguments**:
- `--port`: Specify the port to run Gradio demo (default: 5003)
- `--out_path`: Output directory for saved models (default: `/tmp/xtts_ft/`)
- `--num_epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size for training (default: 4)
- `--grad_acumm`: Gradient accumulation steps (default: 1)
- `--max_audio_length`: Maximum audio length in seconds (default: 11)

---

## 📝 File Overview

- **`xtts_demo.py.py`**: Main script to fine-tune, load, and run the XTTS model all in a gradio web gui.
- **`train_gpt.py`**: Handles the GPT training aspects during fine-tuning.
- **`format_audio_list.py`**: Preprocesses the dataset for training.
- **`export_model()`**: Compresses and exports the fine-tuned model as a `.zip` file.

---

## ✨ Features
- 🔥 Fine-tune XTTS models effectivly on cpu, cuda or even Apple Silicon.
- 📂 Automatically compress and export the best model after fine-tuning.
- 🧠 Leverage model compression to optimize performance.
- 🌍 Supports various languages for training and inference.

---

## ⚠️ Notes
- The apple silicon install was tested on an **M1 Pro** Mac with **16GB RAM** and python 3.10.
  
---

Feel free to contribute, suggest improvements, or raise any issues. Happy fine-tuning! 😎
