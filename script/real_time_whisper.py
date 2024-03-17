import argparse
import os
import sys
from sys import platform

sys.path.append("/home/lucyshi/code/yay_robot/src")
sys.path.append("/home/huzheyuan/Desktop/yay_robot/src")
from aloha_pro.aloha_scripts.audio_utils import AudioTranscriber
import rospy
from std_msgs.msg import String
import time
from pynput import keyboard
from threading import Lock


def main(args):
    # Load / Download model
    model = args.model

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    audio_transcriber = AudioTranscriber()
    # Setup audio
    audio_transcriber.setup_audio(
        model_name=model,
        energy_threshold=args.energy_threshold,
        sample_rate=48000 if "linux" in platform else 16000,
        default_microphone=args.default_microphone if "linux" in platform else None,
    )

    # Cue the user that we're ready to go.
    print("Audio setup completed.\n")

    transcription = []

    transcriber_node = rospy.init_node("transcriber")
    transcriber_publisher = rospy.Publisher("audio_transcription", String, queue_size=1)

    running = True

    def shutdown_hook():
        nonlocal running
        running = False
        print("Shutting down...")

    rospy.on_shutdown(shutdown_hook)

    recording = False

    def on_press(key):
        nonlocal recording
        if key.char == "2":
            # if key == keyboard.Key.esc:
            recording = True

    def on_release(key):
        nonlocal recording
        if key.char == "2":
            # if key == keyboard.Key.esc:
            recording = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    lock = Lock()

    while running:
        if not recording:
            # with lock:
            audio_transcriber.recording = False
            time.sleep(0.1)
            continue

        # with lock:
        audio_transcriber.recording = True
        print("recording command...")
        command = audio_transcriber.transcribe_command(record_timeout, phrase_timeout)
        audio_transcriber.recording = False
        if command:
            transcriber_publisher.publish(command)
            print(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="medium",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
    )
    parser.add_argument(
        "--energy_threshold",
        default=100,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=0.1,
        help="How real-time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=0.5,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="USB PnP Audio Device: Audio",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    main(parser.parse_args())
