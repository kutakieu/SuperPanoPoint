from pathlib import Path
from random import randint

from superpanopoint.datasets.data_synth import SynthData
from superpanopoint.datasets.data_synth.shapes import (Checkerboard, Cube,
                                                       Line, Polygon, Star,
                                                       Stripe)

NUM_SAMPLES = 10
W, H = 512, 512

NUM_MIN_LINES = 3
NUM_MAX_LINES = 10
NUM_MIN_POLYGONS = 10
NUM_MAX_POLYGONS = 10
NUM_MIN_STARS = 1
NUM_MAX_STARS = 3

def main():
    out_dir = Path("data/synthetic/persepective")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SAMPLES):
        synth_data = SynthData(W, H)
        synth_data.add_shapes(Line, randint(NUM_MIN_LINES, NUM_MAX_LINES))
        synth_data.add_shapes(Polygon, randint(NUM_MIN_POLYGONS, NUM_MAX_POLYGONS))
        synth_data.add_shapes(Star, randint(NUM_MIN_STARS, NUM_MAX_STARS))
        synth_data.export(out_dir, str(i))
        synth_data.export_img(out_dir / "with_points" / f"{str(i)}.png", with_points=True)

    for i in range(NUM_SAMPLES, NUM_SAMPLES*2):
        synth_data = SynthData(W, H)
        if i % 3 == 0:
            synth_data.add_shapes(Cube, 1)
        elif i % 3 == 1:
            synth_data.add_shapes(Checkerboard, 1)
        else:
            synth_data.add_shapes(Stripe, 1)
        synth_data.export(out_dir, str(i))
        synth_data.export_img(out_dir / "with_points" / f"{str(i)}.png", with_points=True)

if __name__ == "__main__":
    main()
