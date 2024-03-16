import subprocess
import json

def get_duration(file_path):
    """Get the duration of a media file using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "json", 
        file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(json.loads(result.stdout)['format']['duration'])

# Paths to the input video and audio files
idx = 18
video_path = f"data/act/aloha_bag_audio/episode_{idx}_video.mp4"
audio_path = f"data/act/aloha_bag_audio/episode_{idx}.wav"

# Path to the output file
output_path = f"data/act/aloha_bag_audio/merged/episode_{idx}_with_audio.mp4"

# Calculate the speedup ratio to adjust the audio speed
video_duration = get_duration(video_path)
audio_duration = get_duration(audio_path)
speedup_ratio = audio_duration / video_duration

# Command to combine the video and audio using ffmpeg
command = [
    "ffmpeg",
    "-i", video_path,
    "-filter_complex", f"[1:a]atempo={speedup_ratio}[aout]", # speed up audio
    "-i", audio_path,
    "-map", "0:v", # map video from first input
    "-map", "[aout]", # map audio from filter complex
    "-c:v", "copy",
    "-c:a", "aac",
    "-strict", "experimental",
    output_path
]

# Execute the command
subprocess.run(command, check=True)
