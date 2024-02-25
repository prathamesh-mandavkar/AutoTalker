import gradio as gr
from tts import AudioProcessor
from subtitles import SubtitleGenerator
import torch
import subprocess
import os
import glob
from pydub import AudioSegment
from PIL import Image
import numpy as np
import google.generativeai as genai
import gc
import shutil

gemini_key = os.environ.get('GEMINI_KEY')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-pro')

def remove_special_characters(input_text):
    characters_to_remove = "#*`"
    translation_table = str.maketrans("", "", characters_to_remove)
    cleaned_text = input_text.translate(translation_table)
    return cleaned_text

def lip_sync_audio_and_video(audio_path, source_image_path, result_dir, still, preprocess, enhancer,
                              pose_style, exp_weight, size_of_image):
    command = [
        'python', './SadTalker/inference.py',
        '--driven_audio', str(audio_path),
        '--source_image', str(source_image_path),
        '--result_dir', str(result_dir),
        '--still' if still else '',
        '--preprocess', str(preprocess),
        '--enhancer', str(enhancer),
        '--pose_style', str(pose_style),
        '--expression_scale', str(exp_weight),
        '--size', str(size_of_image),
    ]
    subprocess.run(command)

def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    os.remove(mp3_file)
    return wav_file

def find_latest_mp4_file(folder_path):
    mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))
    if not mp4_files:
        return None
    latest_file = max(mp4_files, key=os.path.getmtime)
    return latest_file.replace('\\', '/')

def generate_output(prompt, script_content, prompt_or_script, user_image, language, voice_gender,
                    pose_style, exp_weight, size_of_image, preprocess_type,
                    is_still_mode, gfpgan_enhancer,subtitle):
    # Cleanup existing files and results folder
    existing_files = ['my_file.mp3', 'enhance_my_file.mp3', 'enhance_my_file.wav']
    for file_path in existing_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    results_folder = './results'
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)

    lang_to_voice_preset = {
        "English": {"male": "v2/en_speaker_7", "female": "v2/en_speaker_9"},
        "Chinese (Simplified)": {"male": "v2/zh_speaker_0", "female": "v2/zh_speaker_4"},
        "French": {"male": "v2/fr_speaker_0", "female": "v2/fr_speaker_1"},
        "German": {"male": "v2/de_speaker_0", "female": "v2/de_speaker_3"},
        "Hindi": {"male": "v2/hi_speaker_5", "female": "v2/hi_speaker_0"},
        "Italian": {"male": "v2/it_speaker_0", "female": "v2/it_speaker_2"},
        "Japanese": {"male": "v2/ja_speaker_2", "female": "v2/ja_speaker_0"},
        "Korean": {"male": "v2/ko_speaker_0", "female": "v2/ko_speaker_4"},
        "Polish": {"male": "v2/pl_speaker_0", "female": "v2/pl_speaker_4"},
        "Portuguese": {"male": "v2/pt_speaker_0", "female": "v2/pt_speaker_4"},
        "Russian": {"male": "v2/ru_speaker_0", "female": "v2/ru_speaker_5"},
        "Spanish": {"male": "v2/es_speaker_0", "female": "v2/es_speaker_8"},
        "Turkish": {"male": "v2/tr_speaker_0", "female": "v2/tr_speaker_4"},
    }

    # Save the user uploaded image without any changes
    user_image_path = "user_image.jpg"
    Image.fromarray(user_image).save(user_image_path)

    ap = AudioProcessor()
    if prompt_or_script == "Prompt":
        if language == "English":
            text = remove_special_characters(model.generate_content(contents=prompt).text)
        else:
            text = model.generate_content(contents=prompt).text
    else:
        text = script_content

    voice_preset = lang_to_voice_preset.get(language, {}).get(voice_gender, "v2/en_speaker_7")

    with torch.inference_mode():
        audio_data = ap.BarkTTS(text, voice_preset)

    file = "my_file.mp3"
    ap.save_audio(audio_data=audio_data, output_file=file, format="mp3")

    torch.cuda.empty_cache()
    gc.collect()

    output_file = f"enhance_{file}"
    ap.enhance_audio(input_file=file, output_file=output_file)

    wav_file = convert_mp3_to_wav(output_file)

    lip_sync_audio_and_video(
        audio_path=wav_file,
        source_image_path=user_image_path,
        result_dir='./results',
        still=is_still_mode,
        preprocess=preprocess_type,
        enhancer='gfpgan' if gfpgan_enhancer else '',
        pose_style=pose_style,
        exp_weight=exp_weight,
        size_of_image=size_of_image,
    )

    torch.cuda.empty_cache()
    gc.collect()

    results_folder = "./results"
    video_filename = find_latest_mp4_file(results_folder)
    if language == "English" and subtitle:
        subtitle_generator = SubtitleGenerator(wav_file, video_filename)
        subtitle_generator.transcribe_audio()
        output_video = subtitle_generator.process_subtitles()

        torch.cuda.empty_cache()
        gc.collect()

        return output_video
    else:
        return video_filename

iface = gr.Interface(fn=generate_output, 
                     inputs=[gr.Textbox(label="Prompt", type="text"),
                             gr.Textbox(label="Script Content", type="text"),
                             gr.Radio(["Prompt", "Script Content"], label="Prompt or Script Content Selection"),
                             gr.Image(label="Source Image"),
                             gr.Radio(["English", "Chinese (Simplified)", "French", "German", "Hindi", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Spanish", "Turkish"], label="Language"),
                             gr.Radio(["male", "female"], label="Voice Gender"),
                             gr.Slider(minimum=0, maximum=45, step=1, label="Pose Style", value=0),
                             gr.Slider(minimum=0, maximum=3, step=0.1, label="Expression Scale", value=1),
                             gr.Radio([256, 512], value=256, label="Face Model Resolution"),
                             gr.Radio(["crop", "resize", "full", "extcrop", "extfull"], value="full", label="Preprocess"),
                             gr.Checkbox(label="Still Mode (Fewer head motion, works with preprocess `full`)"),
                             gr.Checkbox(label="GFPGAN as Face enhancer"),
                             gr.Checkbox(label="Subtitle Support (Currently available only for the English Language)")
                            ],
                     outputs=gr.Video(),
                     title="AutoTalker 🤖🗣️📽️",
                     description="The project focuses on leveraging technology to create new courses, personalize existing ones, and enhance the assessment process, ultimately contributing to the development of 21st-century skills in students.")


iface.launch(share=True)
