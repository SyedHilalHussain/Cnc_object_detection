"""Upload a video clip and get it back with objects (people, vehicles, animals, etc.) detected."""
import os
import subprocess
import time
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from ultralytics import YOLO

MODEL_PATH = "yolo26x.pt"
OUTPUT_ROOT = Path("webapp_output")
OUTPUT_ROOT.mkdir(exist_ok=True)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def reencode_h264(src: Path, dst: Path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def detect(video_path, conf, imgsz, progress=gr.Progress()):
    if video_path is None:
        return None, "Upload a video first."

    model = get_model()
    run_dir = OUTPUT_ROOT / f"run_{int(time.time() * 1000)}"

    progress(0, desc="Running detection...")
    results = model.predict(
        source=video_path,
        conf=conf,
        imgsz=int(imgsz),
        save=True,
        project=str(run_dir),
        name="out",
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    max_per_frame = {}
    n_frames = 0
    save_dir = None
    for r in results:
        n_frames += 1
        if save_dir is None:
            save_dir = Path(r.save_dir)
        frame_counts = {}
        for c in r.boxes.cls.tolist():
            name = model.names[int(c)]
            frame_counts[name] = frame_counts.get(name, 0) + 1
        for name, n in frame_counts.items():
            max_per_frame[name] = max(max_per_frame.get(name, 0), n)

    raw_candidates = list(save_dir.glob("*.avi")) + list(save_dir.glob("*.mp4")) if save_dir else []
    if not raw_candidates:
        return None, "Detection finished but no output video was produced."
    raw_out = raw_candidates[0]

    progress(0.9, desc="Encoding for playback...")
    h264_out = save_dir / "detected_h264.mp4"
    reencode_h264(raw_out, h264_out)
    raw_out.unlink()

    if max_per_frame:
        lines = [f"Frames processed: {n_frames}", "Detected (max seen in a single frame):"]
        for name, n in sorted(max_per_frame.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {name}: {n}")
    else:
        lines = [f"Frames processed: {n_frames}", "No objects detected."]

    return str(h264_out), "\n".join(lines)


with gr.Blocks(title="YOLO26 Video Object Detection") as demo:
    gr.Markdown(
        "# Video Object Detection (YOLO26)\n"
        "Upload a video clip. All COCO classes are detected (people, cars, trucks, animals, etc.) "
        "with bounding boxes drawn, and you get back the annotated video."
    )
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Upload video", sources=["upload"])
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            imgsz = gr.Slider(320, 1280, value=640, step=32, label="Inference image size")
            run_btn = gr.Button("Detect objects", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Detected output")
            summary_out = gr.Textbox(label="Detection summary", lines=10)

    run_btn.click(fn=detect, inputs=[video_in, conf, imgsz], outputs=[video_out, summary_out])


if __name__ == "__main__":
    in_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
    demo.launch(share=in_colab)
