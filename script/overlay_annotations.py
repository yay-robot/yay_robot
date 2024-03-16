import cv2
import json

# Load the JSON file and extract action annotations
def load_annotations(json_path):
    with open(json_path, "r") as file:
        annotations = json.load(file)
    return annotations['annotation']['actionAnnotationList']

# Function to overlay text on video with larger font
def overlay_text_on_video_larger_font(video_path, annotations, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    for annotation in annotations:
        start_frame = int(annotation['start'] * fps)
        end_frame = int(annotation['end'] * fps)
        description = annotation['description']

        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay text if within the annotated time range
            if start_frame <= frame_count < end_frame:
                cv2.putText(frame, description, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)

            out.write(frame)
            frame_count += 1

    # Write remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

# Load annotations and process video
json_path = "/path_to_json_file.json"
video_path = "/path_to_input_video.mp4"
output_video_path = "/path_to_output_video.mp4"

action_annotations = load_annotations(json_path)
overlay_text_on_video_larger_font(video_path, action_annotations, output_video_path)
