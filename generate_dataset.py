from pathlib import Path
from random import choice, randint

from superpanopoint.datasets.data_synth import SynthData
from superpanopoint.datasets.data_synth.shapes import (Checkerboard, Cube,
                                                       Line, Polygon, Star,
                                                       Stripe)

NUM_SAMPLES = 1024
W, H = 512, 512

NUM_MIN_LINES = 3
NUM_MAX_LINES = 10
NUM_MIN_POLYGONS = 10
NUM_MAX_POLYGONS = 10
NUM_MIN_STARS = 1
NUM_MAX_STARS = 3

def main():
    out_dir = Path("data/synthetic/perspective")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SAMPLES):
        synth_data = SynthData(W, H)
        if randint(0, 1) == 0:
            adding_shape = choice([Cube, Checkerboard, Stripe])
            synth_data.add_shapes(adding_shape, 1)

        synth_data.add_shapes(Line, randint(NUM_MIN_LINES, NUM_MAX_LINES))
        synth_data.add_shapes(Polygon, randint(NUM_MIN_POLYGONS, NUM_MAX_POLYGONS))
        synth_data.add_shapes(Star, randint(NUM_MIN_STARS, NUM_MAX_STARS))
        synth_data.export(out_dir, str(i))
        synth_data.export_img(out_dir / "with_points" / f"{str(i)}.png", with_points=True)

if __name__ == "__main__":
    main()
