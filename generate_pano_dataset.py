from pathlib import Path
from random import choice, randint

from superpanopoint.datasets.data_synth import PanoSynthData, SynthData
from superpanopoint.datasets.data_synth.shapes import (Checkerboard, Cube,
                                                       Line, Polygon, Star)
from superpanopoint.datasets.data_synth.shapes.utils import \
    generate_symetric_background

NUM_SAMPLES = 10
W, H = 512, 512
PANO_W, PANO_H = 2048, 1024

NUM_MIN_LINES = 3
NUM_MAX_LINES = 10
NUM_MIN_POLYGONS = 10
NUM_MAX_POLYGONS = 10
NUM_MIN_STARS = 1
NUM_MAX_STARS = 3

def main():
    out_dir = Path("data/synthetic/panorama")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SAMPLES):
        bg_img = generate_symetric_background(W, H)
        synth_data_list = []
        for _ in range(6):
            synth_data = SynthData(W, H, bg_img.copy())
            if randint(0, 1) == 0:
                adding_shape = choice([Cube, Checkerboard])
                synth_data.add_shapes(adding_shape, 1)
            synth_data.add_shapes(Line, randint(NUM_MIN_LINES, NUM_MAX_LINES))
            synth_data.add_shapes(Polygon, randint(NUM_MIN_POLYGONS, NUM_MAX_POLYGONS))
            synth_data.add_shapes(Star, randint(NUM_MIN_STARS, NUM_MAX_STARS))
            synth_data_list.append(synth_data)

        pano_synth_data = PanoSynthData(synth_data_list, W, PANO_W, PANO_H)
        pano_synth_data.export(out_dir, str(i))
        pano_synth_data.export_img(out_dir / "with_points" / f"{str(i)}.png", with_points=True)


if __name__ == "__main__":
    main()
