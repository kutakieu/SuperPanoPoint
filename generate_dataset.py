from pathlib import Path

from superpanopoint.datasets.data_synth import generate_perspective_sample

NUM_SAMPLES = 256
W, H = 256, 256

def main():
    out_dir = Path("data/synthetic/perspective")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SAMPLES):
        synth_data = generate_perspective_sample(W, H)
        synth_data.export(out_dir, str(i))
        synth_data.export_img(out_dir / "with_points" / f"{str(i)}.png", with_points=True)

if __name__ == "__main__":
    main()
