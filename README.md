# AutoTalker 🤖🗣️📽️
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pmking27/AutoTalker/blob/main/Notebook/AutoTalker.ipynb)

## Project Demo Video

Check out the demo video to see AutoTalker in action!

https://github.com/Pmking27/AutoTalker/assets/97112558/59213b80-2613-4999-9e69-9e93d4fda682

## Example

**Input Prompt**

"Explain python and their applications in 30 second"

*Input Image*

 <img src='./examples/image4.jpg' width='380'>

*Output Video*

[https://github.com/Pmking27/AI-Video-generation-Using-Palm-Api-and-SadTalker/assets/97112558/ee86baae-bd2c-4a67-80c2-3768d6f571d6](https://github.com/Pmking27/AutoTalker/assets/97112558/b8a35e28-14ad-42e6-8a7c-e9e28d9bfda5)

## Project Description

### The Apprentice Project (TAP)

In the rapidly evolving landscape of the 21st century, a comprehensive education is paramount for preparing students with the skills necessary to thrive in modern society. The Apprentice Project (TAP) is dedicated to cultivating these essential 21st-century skills among underserved children enrolled in government or low-income private schools.

TAP operates under the umbrella of the MentorMe Foundation, a Section 8 registered company, and is proudly supported by esteemed institutions such as Harvard University, IIM Bangalore, and the Nudge Foundation. As official partners with the Governments of Maharashtra and Delhi, TAP has a significant impact, reaching over 31,000 children through its innovative chatbot.

### Problem Statement

A staggering number of middle and high school students—over 100 million—from low-income communities across India lack critical 21st-century skills, including Social & Emotional Learning (SEL) and Financial Literacy. The traditional exam-centric public education system exacerbates this issue, leading to the alarming statistic that 1 in 2 children graduating from the Indian education system is deemed unemployable due to the absence of these crucial skills.

- **Financial Literacy Statistics:**
  - Only 16.7% of Indian teenage students possess basic financial literacy.
  - A mere 27% of the Indian adult population is considered financially literate.

### UN Sustainable Development Goals (SDGs)

TAP aligns its mission with several UN SDGs:

- **Goal 1:** No Poverty
- **Goal 2:** Zero Hunger
- **Goal 3:** Good Health and Well-being
- **Goal 4:** Quality Education
- **Goal 8:** Decent Work and Economic Growth

### Current System

The Apprentice Project (TAP), operating under the MentorMe Foundation, empowers underserved students through TAP Buddy—an artificial intelligence-powered WhatsApp chatbot. TAP Buddy offers video-based electives, guiding students through independent projects using personalized (ML-learned) and AI bot-based nudges and content. Self-learning project videos foster skills such as creativity, confidence, self-awareness, communication, and problem-solving, breaking mental barriers and instilling a growth mindset.

### Challenges and Innovations

As the usage of TAP's chatbot continues to grow, the project faces challenges and seeks innovative solutions:

1. **Course Creation:** Leveraging AI to generate content across various electives such as coding and visual arts, aiming to overcome limitations in bulk video creation due to manual time constraints.

2. **Personalized Learning:** Employing AI to create personalized coding tutorials or art project guides tailored to individual learning styles and skill levels. Advanced ML/Open AI analysis adapts content based on a learner's progress, ensuring a customized learning experience.

3. **Content Creation:** Utilizing AI to generate code snippets, templates, or design ideas for art projects, guiding students at their skill levels and suggesting exploration options.

4. **Artistic Exploration:** Recommending techniques and styles based on a child's skill level, broadening artistic horizons by comparing their work to famous artists or art movements.

5. **Creative Coding:** Using AI to brainstorm ideas and provide inspiration for innovative and artistic coding projects.

## Approach and Solution

My approach to addressing the challenges faced by TAP involves leveraging cutting-edge technologies, including natural language processing (NLP), artificial intelligence (AI), and machine learning (ML), to develop AutoTalker—a component of TAP aimed at enhancing the educational experience for students.

AutoTalker utilizes advanced AI models and libraries, such as Suno Bark TTS for text-to-speech conversion, Google's generative AI Python SDK (Gemini Pro) for text generation, and SadTalker for lip-syncing audio with facial movements in videos. By integrating these technologies, AutoTalker enables the creation of engaging and informative video content from text prompts and images.

Furthermore, the project incorporates features like personalized learning, content creation assistance, and language support to cater to diverse learning needs and preferences. By harnessing the power of AI, AutoTalker empowers educators and students alike to access high-quality educational content tailored to their individual requirements, thereby fostering the development of essential 21st-century skills.

Through this innovative solution, TAP aims to revolutionize the education landscape, bridging the gap in access to quality learning resources and empowering students from underserved communities to realize their full potential in the digital age.
## Table of Contents
- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About

The project focuses on leveraging technology to create new courses, personalize existing ones, and enhance the assessment process, ultimately contributing to the development of 21st-century skills in students. AutoTalker, a component of TAP, showcases the capabilities of AI in generating lip-synced videos from text prompts and images, enhancing the overall educational experience for students. 

It utilizes several libraries, including:

- **Suno Bark TTS:** A text-to-speech conversion library used for generating audio from text prompts.
- **pydub:** An audio manipulation library for handling audio files and formats.
- **google.generativeai (Gemini Pro):** Google's generative AI Python SDK utilized for text generation.
- **SadTalker:** A lip-syncing model used for synchronizing audio with facial movements in videos.
- **Openai Whisper:** A library for speech-to-text conversion, enabling the customization of voice characteristics.
- **Spotify Pedalboard:** An audio enhancement library for improving the quality and effects of audio files.
- **MoviePy:** A video editing library that facilitates video processing and editing tasks.
- **PyTorch:** A deep learning framework used for various machine learning tasks, including SadTalker's functionality.
- **FFmpeg:** A multimedia framework used for handling multimedia data such as audio and video files.
- **Hugging Face Transformers:** A library that provides pre-trained models and various utilities for natural language processing tasks.
- **BetterTransformer:** A production-ready fast path to accelerate the deployment of Transformer models with high performance on CPU and GPU. The fast path feature works transparently for models based either directly on PyTorch core nn.
- **NumPy:** A powerful numerical computation library for handling large, multi-dimensional arrays and matrices.
- **Gradio:** A user-friendly library for creating customizable UI components around machine learning models, enabling easy deployment and interaction with models through web interfaces.

## Features

- **Text-to-Speech Conversion:** Utilizes Suno Bark TTS to convert text prompts into audio files (WAV format).
- **Audio Manipulation:** Employs pydub for audio manipulation tasks, enhancing audio quality and applying desired effects.
- **Generative AI Text:** Leverages Google's generative AI Python SDK (Gemini Pro) for text generation, providing diverse and contextually relevant prompts.
- **Lip Syncing:** Integrates SadTalker, a lip-syncing model, to synchronize generated audio with facial movements in videos.
- **Speech-to-Text Conversion:** Incorporates Openai Whisper for speech-to-text conversion, allowing customization of voice characteristics.
- **Audio Enhancement:** Utilizes Spotify Pedalboard to enhance and apply effects to audio files, improving overall audio quality.
- **Video Editing:** Implements MoviePy, a video editing library, for video processing and editing tasks, including the creation of final lip-synced videos.
- **Deep Learning Framework:** Leverages PyTorch for its deep learning capabilities, essential for executing SadTalker's functionality.
- **Multimedia Handling:** Uses FFmpeg, a multimedia framework, for handling multimedia data such as audio and video files during processing.
- **Natural Language Processing:** Integrates Hugging Face Transformers, offering pre-trained models and utilities for natural language processing tasks.
- **Fast Path for Transformer Models:** Incorporates BetterTransformer, a production-ready fast path for accelerated deployment of Transformer models on both CPU and GPU.
- **Numerical Computation:** Relies on NumPy for powerful numerical computation, particularly for handling large, multi-dimensional arrays and matrices.
- **User-Friendly UI Components:** Integrates Gradio, a user-friendly library, to create customizable UI components around machine learning models, facilitating easy deployment and interaction through web interfaces.
- **Language Support:** Supports multiple languages, including English, Chinese (Simplified), French, German, Hindi, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, and Turkish.
- **Subtitle Support:** Currently available only for the English language.

These features collectively contribute to the generation of lip-synced videos from input text prompts and images, with support for various languages and subtitles in English.


## Getting Started

### Prerequisites

- [Python 3.10.6](https://www.python.org/downloads/release/python-3106/)
- [API key](https://ai.google.dev/) from Google AI.
- [ffmpeg](https://ffmpeg.org/download.html) installed.
- [PyTorch](https://pytorch.org/) installed. Ensure your system supports CUDA.
- [ImageMagick](https://imagemagick.org/script/download.php) installed. This is required for MoviePy.
- [SadTalker](https://github.com/OpenTalker/SadTalker) installed.

  **Note:** Ensure your GPU has a minimum of 4 GB VRAM with support for CUDA.

### Installation
1. **Install Python 3.10.6:**
   - Download and install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/). Note that versions 3.11 and 3.12 are not supported.

2. **Install ffmpeg:**
   - Follow the appropriate instructions for your system.
  
3. **Install ImageMagick:**
   - Download and install [ImageMagick](https://imagemagick.org/script/download.php).

4. **Clone the AutoTalker repository:**
   ```bash
   git clone https://github.com/Pmking27/AutoTalker
   cd AutoTalker
   ```

5. **Download SadTalker with Models and Weights:**
   ```bash
   python download_models.py
   ```
   
   Run the above command and wait until it shows "Downloads completed." This will download SadTalker along with the required models and weights.

6. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

7. **Activate the virtual environment:**
   - **On Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     .\venv\Scripts\activate
     ```

8. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

9. **Install PyTorch with CUDA:**
   ```bash
   pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   ```

Now, you have successfully set up the environment for the project, ensuring your GPU meets the specified requirements.

## Usage

### Project Structure

The project has the following structure:

```
.
├── checkpoints    # Model checkpoints (SadTalker)
│   ├── _MACOSX
│   ├── mapping_00109-model.pth.tar
│   ├── mapping_00229-model.pth.tar
│   ├── SadTalker_V0.0.2_256.safetensors
│   └── SadTalker_V0.0.2_512.safetensors
│── gfpgan_weights  # Weights for GFPGAN enhancer
│   ├── _MACOSX
│   ├── alignment_WFLW_4HG.pth
│   ├── detection_Resnet50_Final.pth
│   ├── GFPGANv1.4.pth
│   └── parsing_parsenet.pth
├── SadTalker   # Folder containing SadTalker code
│   ├── app_sadtalker.py
│   ├── cog.yaml
│   ├── inference.py
│   ├── launcher.py
│   ├── LICENSE
│   ├── predict.py
│   ├── quick_demo.ipynb
│   ├── README.md
│   ├── req.txt
│   ├── requirements.txt
│   ├── requirements3d.txt
│   ├── webui.bat
│   └── webui.sh
├── venv                 # Virtual environment folder
├── download_models.py   # Models download script
├── main.py              # Main Python script
├── requirements.txt     # All required dependencies list txt file
├── subtitles.py         # Audio Enhacing and subtitles creation script
└── tts.py               # Text To Speech into .wav file creation script
```

### Steps to Running AutoTalker and Opening Gradio Web UI:

1. **Activate Virtual Environment:**
   - Activate the virtual environment created earlier.

2. **Configure GEMINI PRO API Key:**
   - Open the `main.py` file.
   - Locate the line: `genai.configure(api_key="add your key here")`.
   - Replace `"add your key here"` with your actual GEMINI PRO API key.

3. **Run Main Script and Gradio Web UI:**
   - Copy the provided Gradio code (`iface.launch()` part) from the script.

4. **Run AutoTalker and Launch Gradio:**
   - In the same terminal where your virtual environment is active, run the AutoTalker script along with the Gradio web UI.
   ```bash
   python main.py
   ```

5. **Access Gradio Web UI:**
   - After running the script, Gradio will provide a link (usually localhost) where the web UI is hosted. Open that link in your web browser.

6. **Explore the Interface:**
   - You will now have access to the Gradio web UI interface.
   - Interact with the provided input components such as textboxes, radio buttons, sliders, and image upload options.

7. **Submit and Wait:**
   - Click on the "Launch Interface" or similar button to submit your input.
   - Gradio will process your input, generate the output, and display the results in the web UI.

8. **Review Output:**
   - The output may be a video with or without subtitles, depending on your configuration.

9. **Explore Subtitles (If Enabled):**
   - If you enabled subtitle support, explore the subtitles generated for the video.

10. **Repeat and Experiment:**
    - Feel free to experiment with different inputs, prompts, and parameters to generate various outputs.

11. **Close Gradio UI:**
    - Once you're done, close the Gradio web UI.

By following these combined steps, you can seamlessly run AutoTalker, interact with the Gradio web UI, and experience the generated lip-synced videos.



## Contributing

We appreciate your interest in contributing to our project! To ensure a smooth and collaborative experience, please follow these guidelines:

1. **Fork the Repository:**
   - Start by forking this repository to your GitHub account.

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/AutoTalker.git
   ```

3. **Create a Branch:**
   - For each contribution, create a new branch with a descriptive name.
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes:**
   - Implement your enhancements or fixes. Ensure your changes align with the project's goals.

5. **Commit Changes:**
   - Commit your changes with clear and concise commit messages.
   ```bash
   git commit -m "Add your commit message here"
   ```

6. **Push Changes:**
   - Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request:**
   - Open a pull request from your forked repository to the main repository.
   - Provide detailed information about your changes, outlining the purpose and impact.

8. **Review and Collaborate:**
   - Engage in discussions, respond to feedback, and collaborate with the community to refine your contribution.

9. **Squash Commits (if needed):**
   - If your pull request contains multiple commits, consider squashing them into a single, well-structured commit.

10. **Merge:**
    - Once your pull request is approved, it will be merged into the main repository.

11. **Areas Needing Help: Human-Like TTS Implementation**

   If you're interested in making a significant impact, consider contributing to the implementation of Human-Like Text-to-Speech (TTS) for a diverse set of languages, including Indian regional languages. Focus on enhancing TTS capabilities for both male and female voices.

   ### Supported Languages for Human-Like TTS Implementation:

   - Arabic (ar)
   - Bengali (bn)
   - Bulgarian (bg)
   - Croatian (hr)
   - Czech (cs)
   - Danish (da)
   - Dutch (nl)
   - Estonian (et)
   - Finnish (fi)
   - Greek (el)
   - Hebrew (iw)
   - Hungarian (hu)
   - Indonesian (id)
   - Latvian (lv)
   - Lithuanian (lt)
   - Norwegian (no)
   - Romanian (ro)
   - Serbian (sr)
   - Slovak (sk)
   - Slovenian (sl)
   - Swahili (sw)

   ### Additional Focus on Indian Regional Languages:

   Given the diverse linguistic landscape in India, contributions to support Indian regional languages in TTS are highly valued. These languages may include, but are not limited to:

   - Hindi
   - Tamil
   - Telugu
   - Kannada
   - Malayalam
   - Punjabi
   - Gujarati
   - Marathi
   - Bengali
   - Odia
   - Assamese
   - Urdu

   Your efforts in implementing TTS for these languages will significantly contribute to making educational content accessible to a broader audience, particularly in regions with diverse linguistic backgrounds.

   Thank you for considering these important contributions to the Human-Like TTS implementation! Your work will play a vital role in making educational content inclusive and accessible to learners from various linguistic backgrounds. 🌐🔊
## License

This project is licensed under the MIT License.

## Acknowledgements

This project acknowledges the following open-source projects and their contributors:

- **[Google AI Python SDK](https://github.com/google/generative-ai-python):**
  The Google AI Python SDK enables developers to use Google's state-of-the-art generative AI models (like Gemini and PaLM) to build AI-powered features and applications.

- **[SadTalker](https://github.com/OpenTalker/SadTalker):**
  [CVPR 2023] SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation. A project by OpenTalker.

- **[Pedalboard](https://github.com/spotify/pedalboard):**
  A Python library for working with audio, developed by Spotify.

- **[Whisper](https://github.com/openai/whisper):**
  Robust Speech Recognition via Large-Scale Weak Supervision, an open-source project by OpenAI.

- **[Transformers by Hugging Face](https://github.com/huggingface/transformers):**
  🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.

- **[Accelerate by Hugging Face](https://github.com/huggingface/accelerate):**
  A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.

- **[Optimum by Hugging Face](https://github.com/huggingface/optimum):**
  🚀 Accelerate training and inference of 🤗 Transformers and 🤗 Diffusers with easy-to-use hardware optimization tools.

- **[Bark by Suno AI](https://github.com/suno-ai/bark):**
  🔊 Text-Prompted Generative Audio Model.

- **[PyTorch](https://github.com/pytorch/pytorch):**
  Tensors and Dynamic neural networks in Python with strong GPU acceleration.

These projects have significantly contributed to the development and functionality of AutoTalker, and we extend our gratitude to their respective developers and maintainers.



