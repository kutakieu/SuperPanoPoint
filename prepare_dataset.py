from pathlib import Path
from random import randint

from superpanopoint.datasets.data_synth.shapes import Line, Polygon, SynthData

NUM_SAMPLES = 20
W, H = 1024, 512

NUM_MIN_LINES = 3
NUM_MAX_LINES = 10
NUM_MIN_POLYGONS = 3
NUM_MAX_POLYGONS = 10


def main():
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SAMPLES):
        synth_data = SynthData(W, H)
        synth_data.add_shapes(Line, randint(NUM_MIN_LINES, NUM_MAX_LINES))
        synth_data.add_shapes(Polygon, randint(NUM_MIN_POLYGONS, NUM_MAX_POLYGONS))
        synth_data.export(out_dir, str(i))

if __name__ == "__main__":
    main()
