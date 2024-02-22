# AutoTalker 🤖🗣️📽️

## Example

**Input Prompt**

"Explain python and their applications in 30 second"

*Input Image*

 <img src='./examples/image4.jpg' width='380'>

*Output Video*

[https://github.com/Pmking27/AI-Video-generation-Using-Palm-Api-and-SadTalker/assets/97112558/ee86baae-bd2c-4a67-80c2-3768d6f571d6](https://github.com/Pmking27/AutoTalker/assets/97112558/b8a35e28-14ad-42e6-8a7c-e9e28d9bfda5)


# Project Description

## The Apprentice Project (TAP)

Living in the 21st century requires a well-rounded education that equips students with essential skills for success. The Apprentice Project (TAP) is dedicated to fostering 21st-century skills among underserved children in government or low-income private schools.

TAP operates under the MentorMe Foundation, a Section 8 registered company. Supported by prestigious institutions such as Harvard University, IIM Bangalore, and the Nudge Foundation, TAP is an official partner with the Governments of Maharashtra and Delhi, impacting over 31,000 children through its innovative chatbot.

### Problem Statement

Over 100 million middle and high school students from low-income communities in India lack critical 21st-century skills, including Social & Emotional Learning and Financial Literacy. The traditional exam-focused public education system contributes to the alarming fact that 1 in 2 children passing through the Indian education system is deemed unemployable due to the absence of these crucial skills.

- **Financial Literacy Stats:**
  - Only 16.7% of Indian teenage students have basic financial literacy.
  - Only 27% of the Indian adult population is financially literate.

### UN Sustainable Development Goals (SDGs)

TAP aligns with several UN SDGs:

- **Goal 1:** No Poverty
- **Goal 2:** Zero Hunger
- **Goal 3:** Good Health and Well-being
- **Goal 4:** Quality Education
- **Goal 8:** Decent Work and Economic Growth

### Current System

The Apprentice Project(TAP), under MentorMe Foundation, empowers underserved students through TAP Buddy—an artificial intelligence-powered WhatsApp chatbot. TAP Buddy offers video-based electives, guiding students through independent projects using personalized (ML-learned) and (AI) bot-based nudges and content. Self-learning project videos foster skills such as creativity, confidence, self-awareness, communication, and problem-solving, breaking mental barriers and instilling a growth mindset.

### Challenges and Innovations

As the number of students utilizing TAP's chatbot grows, the project faces challenges and seeks innovative solutions:

1. **Course Creation:** Utilizing AI to generate content in various electives such as coding and visual arts, aiming to overcome limitations in bulk video creation due to manual time constraints.

2. **Personalized Learning:** Employing AI to create personalized coding tutorials or art project guides tailored to individual learning styles and skill levels. Advanced ML/Open AI analysis adapts content based on a learner's progress, ensuring a customized learning experience.

3. **Content Creation:** Leveraging AI to generate code snippets, templates, or design ideas for art projects, guiding students at their skill levels and suggesting exploration options.

4. **Artistic Exploration:** Recommending techniques and styles based on a child's skill level, broadening artistic horizons by comparing their work to famous artists or art movements.

5. **Creative Coding:** Utilizing AI to brainstorm ideas and provide inspiration for innovative and artistic coding projects.

The project focuses on leveraging technology to create new courses, personalize existing ones, and enhance the assessment process, ultimately contributing to the development of 21st-century skills in students.
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
- [SadTalker](https://github.com/OpenTalker/SadTalker) installed.

  **Note:** Ensure your GPU has a minimum of 4 GB VRAM with support for CUDA.

### Installation
1. **Install Python 3.10.6:**
   - Download and install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/). Note that versions 3.11 and 3.12 are not supported.

2. **Install ffmpeg:**
   - Follow the appropriate instructions for your system.

3. **Clone the AutoTalker repository:**
   ```bash
   git clone https://github.com/Pmking27/AutoTalker
   cd AutoTalker
   ```

4. **Download SadTalker with Models and Weights:**
   ```bash
   python download_models.py
   ```
   
   Run the above command and wait until it shows "Downloads completed." This will download SadTalker along with the required models and weights.

5. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

6. **Activate the virtual environment:**
   - **On Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     .\venv\Scripts\activate
     ```

7. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

8. **Install PyTorch with CUDA:**
   - Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA support based on your system.

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

### Steps

1. **Activate Virtual Environment:**
   - Activate the virtual environment created earlier.

2. **Configure GEMINI PRO API Key:**
   - Open the `main.py` file.
   - Locate the line: `genai.configure(api_key="add your key here")`.
   - Replace `"add your key here"` with your actual GEMINI PRO API key.

3. **Run the Main Script:**
   ```bash
   python main.py
   ```

4. **Customize:**
   - Modify the prompt in the `main()` function to generate text about different topics.
   - Adjust audio and image paths, as well as the output directory, as needed.
   - Explore different options for preprocessing and enhancement within the `lip_sync_audio_and_video()` function.

5. **Review Results:**
   - The lip-synced video will be generated and saved in the specified output directory.

Feel free to explore the provided file structure, customize the project, and configure the GEMINI PRO API key as needed!



## Contributing

We welcome contributions to improve and enhance this project! If you would like to contribute, please follow these guidelines:

1. **Fork the Repository:**
   - Fork this repository to your GitHub account.

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/AutoTalker.git
   ```

3. **Create a Branch:**
   - Create a new branch for your contributions.
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes:**
   - Implement your changes and improvements.

5. **Commit Changes:**
   - Commit your changes with descriptive commit messages.
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
   - Provide a clear description of your changes.

8. **Review and Collaborate:**
   - Participate in discussions and address feedback on your pull request.

9. **Squash Commits (if needed):**
   - If there are multiple commits, consider squashing them into a single, concise commit.

10. **Merge:**
    - Once approved, your pull request will be merged.

11. **Areas Needing Help:**
    - If you are looking to contribute but need guidance, we have specific areas where help is appreciated:
      - **Human-Like TTS Implementation:**
        - Assistance is needed for implementing human-like TTS for both male and female voices.
      - **Prompt Engineering:**
        - Suggestions and contributions for better prompt engineering.
      - **Feature Suggestions:**
        - Share new feature ideas or improvements that could enhance the project.

Thank you for contributing! 🚀


## License

This project is licensed under the MIT License.

## Acknowledgements

This project acknowledges the following open-source projects and their contributors:

- **[Google AI Python SDK](https://github.com/google/generative-ai-python):**
  The Google AI Python SDK enables developers to use Google's state-of-the-art generative AI models (like Gemini and PaLM) to build AI-powered features and applications.

- **[SadTalker](https://github.com/OpenTalker/SadTalker):**
  [CVPR 2023] SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation. A project by OpenTalker.


