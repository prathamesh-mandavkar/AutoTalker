import subprocess
import os
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as palm
palm.configure(api_key="AIzaSyAizYXZa8F6DV-_J6_ava7iSb7-aCD2c6I")

def text_to_audio(text, audio_path):
    tts = gTTS(text, lang='en', slow=False)
    tts.save(audio_path + ".mp3")  # Save as MP3 initially

    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(audio_path + ".mp3")
    audio.export(audio_path + ".wav", format="wav")

    # Remove the temporary MP3 file
    os.remove(audio_path + ".mp3")

def remove_special_characters(input_text):
    # Define the characters to be removed
    characters_to_remove = "#*`"

    # Create a translation table
    translation_table = str.maketrans("", "", characters_to_remove)

    # Use translate to remove the specified characters
    cleaned_text = input_text.translate(translation_table)

    return cleaned_text

def lip_sync_audio_and_video(audio_path, source_image_path, result_dir, still=False, preprocess='full', enhancer='gfpgan'):
    command = [
        'python310', './SadTalker/inference.py',
        '--driven_audio', audio_path,
        '--source_image', source_image_path,
        '--result_dir', result_dir,
        '--still' if still else '',
        '--preprocess', preprocess,
        '--enhancer', enhancer
    ]

    subprocess.run(command)

if __name__ == "__main__":
    # prompt = "Create a comprehensive and engaging script for a Python tutorial video, covering key concepts, syntax, examples, and practical applications. Ensure the script is clear, informative, and suitable for beginners."
    prompt = "Explain python and their applications in 30 second"
    
    text = remove_special_characters(palm.generate_text(prompt=prompt).result)
    print(text)
    text_to_audio(text,'audio')

    lip_sync_audio_and_video(
    audio_path='audio.wav',
    source_image_path='image.jpg',
    result_dir='./results',
    still=True,
    preprocess='full',
    enhancer='gfpgan'
)


