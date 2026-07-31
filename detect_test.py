from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.predict(
    source="6387-191695740_medium.mp4",
    classes=[0],  # person only
    conf=0.25,
    save=True,
    project="runs",
    name="detect_test",
    exist_ok=True,
    stream=True,
)

counts = []
for r in results:
    counts.append(len(r.boxes))

print(f"frames processed: {len(counts)}")
print(f"person count per frame -> min={min(counts)} max={max(counts)} avg={sum(counts)/len(counts):.1f}")
