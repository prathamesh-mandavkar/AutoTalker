import os
import whisper
from moviepy.editor import TextClip, CompositeVideoClip, VideoFileClip, ColorClip

class SubtitleGenerator:
    def __init__(self, audio_filename, video_filename, model_path="medium"):
        self.audio_filename = audio_filename
        self.video_filename = video_filename
        self.model_path = model_path
        self.IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe')
        self.frame_size = None
        self.linelevel_subtitles = None
        self.all_linelevel_splits = None

    def load_audio_model(self):
        return whisper.load_model(self.model_path)

    def transcribe_audio(self):
        model = self.load_audio_model()
        result = model.transcribe(self.audio_filename, word_timestamps=True)
        wordlevel_info = []

        for each in result['segments']:
            words = each['words']
            for word in words:
                wordlevel_info.append({'word': word['word'].strip(), 'start': word['start'], 'end': word['end']})

        self.linelevel_subtitles = self.split_text_into_lines(wordlevel_info)

    def create_caption(self, textJSON, font="Helvetica-Bold", color='white', highlight_color='yellow',
                       stroke_color='black', stroke_width=1.5):
        wordcount = len(textJSON['textcontents'])
        full_duration = textJSON['end']-textJSON['start']

        word_clips = []
        xy_textclips_positions =[]

        x_pos = 0
        y_pos = 0
        line_width = 0  # Total width of words in the current line
        frame_width = self.frame_size[0]
        frame_height = self.frame_size[1]

        x_buffer = frame_width*1/10

        max_line_width = frame_width - 2 * (x_buffer)

        fontsize = int(frame_height * 0.075) #7.5 percent of video height

        space_width = ""
        space_height = ""

        for index,wordJSON in enumerate(textJSON['textcontents']):
            duration = wordJSON['end']-wordJSON['start']
            word_clip = TextClip(wordJSON['word'], font = font,fontsize=fontsize, color=color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(textJSON['start']).set_duration(full_duration)
            word_clip_space = TextClip(" ", font = font,fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
            word_width, word_height = word_clip.size
            space_width,space_height = word_clip_space.size
            if line_width + word_width+ space_width <= max_line_width:
                    # Store info of each word_clip created
                    xy_textclips_positions.append({
                        "x_pos":x_pos,
                        "y_pos": y_pos,
                        "width" : word_width,
                        "height" : word_height,
                        "word": wordJSON['word'],
                        "start": wordJSON['start'],
                        "end": wordJSON['end'],
                        "duration": duration
                    })

                    word_clip = word_clip.set_position((x_pos, y_pos))
                    word_clip_space = word_clip_space.set_position((x_pos+ word_width, y_pos))

                    x_pos = x_pos + word_width+ space_width
                    line_width = line_width+ word_width + space_width
            else:
                    # Move to the next line
                    x_pos = 0
                    y_pos = y_pos+ word_height+10
                    line_width = word_width + space_width

                    # Store info of each word_clip created
                    xy_textclips_positions.append({
                        "x_pos":x_pos,
                        "y_pos": y_pos,
                        "width" : word_width,
                        "height" : word_height,
                        "word": wordJSON['word'],
                        "start": wordJSON['start'],
                        "end": wordJSON['end'],
                        "duration": duration
                    })

                    word_clip = word_clip.set_position((x_pos, y_pos))
                    word_clip_space = word_clip_space.set_position((x_pos+ word_width , y_pos))
                    x_pos = word_width + space_width


            word_clips.append(word_clip)
            word_clips.append(word_clip_space)


        for highlight_word in xy_textclips_positions:

            word_clip_highlight = TextClip(highlight_word['word'], font = font,fontsize=fontsize, color=highlight_color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
            word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
            word_clips.append(word_clip_highlight)

        return word_clips,xy_textclips_positions

    def process_subtitles(self):
        input_video = VideoFileClip(self.video_filename)
        self.frame_size = input_video.size
        self.all_linelevel_splits = []

        for line in self.linelevel_subtitles:
            out_clips, positions = self.create_caption(line)

            max_width = 0
            max_height = 0

            for position in positions:
                x_pos, y_pos = position['x_pos'], position['y_pos']
                width, height = position['width'], position['height']

                max_width = max(max_width, x_pos + width)
                max_height = max(max_height, y_pos + height)

            color_clip = ColorClip(size=(int(max_width * 1.1), int(max_height * 1.1)),
                                color=(64, 64, 64))
            color_clip = color_clip.set_opacity(.6)
            color_clip = color_clip.set_start(line['start'] - 0.1).set_duration(line['end'] - line['start'] + 0.2)

            clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)
            clip_to_overlay = clip_to_overlay.set_position("bottom")

            self.all_linelevel_splits.append(clip_to_overlay)


        final_video = CompositeVideoClip([input_video] + self.all_linelevel_splits)

        final_video = final_video.set_audio(input_video.audio)

        final_video.write_videofile("output.mp4", fps=24, codec="libx264", audio_codec="aac")

        return "output.mp4"


    def split_text_into_lines(self, data):
        MaxChars = 30
        MaxDuration = 2.5
        MaxGap = 1.5

        subtitles = []
        line = []
        line_duration = 0
        line_chars = 0

        for idx, word_data in enumerate(data):
            word = word_data["word"]
            start = word_data["start"]
            end = word_data["end"]

            line.append(word_data)
            line_duration += end - start

            temp = " ".join(item["word"] for item in line)

            new_line_chars = len(temp)

            duration_exceeded = line_duration > MaxDuration
            chars_exceeded = new_line_chars > MaxChars

            if idx > 0:
                gap = word_data['start'] - data[idx - 1]['end']
                maxgap_exceeded = gap > MaxGap
            else:
                maxgap_exceeded = False

            if duration_exceeded or chars_exceeded or maxgap_exceeded:
                if line:
                    subtitle_line = {
                        "word": " ".join(item["word"] for item in line),
                        "start": line[0]["start"],
                        "end": line[-1]["end"],
                        "textcontents": line
                    }
                    subtitles.append(subtitle_line)
                    line = []
                    line_duration = 0
                    line_chars = 0

        if line:
            subtitle_line = {
                "word": " ".join(item["word"] for item in line),
                "start": line[0]["start"],
                "end": line[-1]["end"],
                "textcontents": line
            }
            subtitles.append(subtitle_line)

        return subtitles


