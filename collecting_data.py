from pytubefix import YouTube
import cv2
import os
import re

def download_youtube(url, output_path="video.mp4", po_token=None):
    if po_token:
        yt = YouTube(url, use_po_token=True, po_token=po_token)
    else:
        yt = YouTube(url)

    stream = yt.streams.filter(progressive=True, file_extension="mp4").get_highest_resolution()
    stream.download(filename=output_path)

    print(f"[INFO] Downloaded → {output_path}")
    return output_path


def get_next_sequence_index(out_dir):
    if not os.path.exists(out_dir):
        return 1

    seqs = [d for d in os.listdir(out_dir) if re.match(r"seq_\d{5}$", d)]
    if not seqs:
        return 1
    
    numbers = [int(d.split("_")[1]) for d in seqs]
    return max(numbers) + 1


def extract_sequences(
    video_path,
    out_dir="sequences",
    seq_frames=30,
    interval_sec=30,
    resize=(224, 224)
):
    os.makedirs(out_dir, exist_ok=True)

    seq_idx = get_next_sequence_index(out_dir)
    print(f"[INFO] Starting new sequences from seq_{seq_idx:05d}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"[INFO] FPS: {fps}, Duration: {duration:.2f} sec")

    start_times = list(range(0, int(duration), interval_sec))

    for start_sec in start_times:
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        seq_folder = os.path.join(out_dir, f"seq_{seq_idx:05d}")
        os.makedirs(seq_folder, exist_ok=True)

        print(f"[INFO] Extracting seq_{seq_idx:05d} (from {start_sec}s)")

        for f in range(seq_frames):
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, resize)

            filename = f"{f+1:05d}.jpg"
            cv2.imwrite(os.path.join(seq_folder, filename), resized)

        seq_idx += 1

    cap.release()
    print("[INFO] Done extracting sequences.")

##------train data------
##url = "https://www.youtube.com/watch?v=RseCk0mwFE8&t=11s"
##url = "https://www.youtube.com/watch?v=5FjqTsc9gX0"
##url = "https://www.youtube.com/watch?v=171gQSf_CPI"
##url = "https://www.youtube.com/watch?v=HPEx4XQXE9Q"
##url = "https://www.youtube.com/watch?v=c5Rl3-oyZQc"

##------validation data------
##url = "https://www.youtube.com/watch?v=M5uCoyQe60s"
##url = "https://www.youtube.com/watch?v=KgH2FQwZXG0"

##------test data------
##url = "https://www.youtube.com/watch?v=AQRxDt9jc1U"

url = "url"
video_path = download_youtube(url, "downloaded.mp4")


extract_sequences(video_path, out_dir="Data/test/sequences", seq_frames=30, interval_sec=30)