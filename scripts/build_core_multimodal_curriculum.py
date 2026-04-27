#!/usr/bin/env python3
"""
Build an expanded multimodal curriculum for Core training.

Prior version (2026-04-23): 60 pairs, minimal coverage. 500-step training
on that curriculum produced color-concept awareness but not individual
color differentiation (model collapsed to always saying "blue" or "black").

This version targets ~400 pairs with deep per-color signal:
  - 12 colors × 8 shade variants × 3 phrased prompts = 288 color pairs
  - 8 shapes × 3 colors × 2 prompts                   = 48 shape pairs
  - 20 color+shape combos × 2 prompts                 = 40 composition pairs
  - 20 single-word text overlays                      = 20 OCR pairs
  - 10 multi-object scenes                            = 10 scene pairs
  → 406 total vision pairs over ~200 unique images

Each color has 8 shade variants to prevent the model from memorizing a
single RGB triple as "red". Prompts are phrased three different ways so
the instruction-following pattern learned isn't tied to one template.

Images saved to knowledge/curricula/core-multimodal/images/
JSONL saved to knowledge/curricula/core-multimodal/vision_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("curriculum")

ROOT = Path("/gaia/GAIA_Project/knowledge/curricula/core-multimodal")
IMAGES_DIR = ROOT / "images"
OUTPUT_JSONL = ROOT / "vision_pairs.jsonl"

# ── Color palette — 12 colors, 8 shade variants each ───────────────────────
# Shades span dark → mid → light so the model learns "red" is a range,
# not one specific RGB triple.
COLORS = {
    "red":    [(120, 10, 10), (160, 20, 20), (200, 30, 30), (220, 40, 40),
               (240, 60, 60), (220, 80, 80), (200, 100, 100), (180, 60, 60)],
    "green":  [(10, 80, 10), (20, 120, 20), (30, 150, 30), (40, 180, 40),
               (60, 200, 60), (80, 220, 80), (100, 200, 100), (40, 160, 40)],
    "blue":   [(10, 10, 120), (20, 20, 160), (30, 30, 200), (40, 40, 220),
               (60, 60, 240), (100, 100, 220), (60, 100, 220), (20, 60, 180)],
    "yellow": [(180, 180, 20), (200, 200, 40), (220, 220, 60), (240, 230, 80),
               (250, 240, 100), (240, 220, 40), (230, 210, 30), (220, 200, 50)],
    "orange": [(220, 120, 20), (240, 140, 30), (240, 160, 60), (230, 130, 40),
               (250, 150, 50), (220, 110, 40), (240, 130, 70), (230, 140, 50)],
    "purple": [(100, 20, 140), (130, 30, 180), (150, 40, 200), (170, 60, 220),
               (180, 80, 210), (140, 50, 190), (120, 30, 160), (160, 70, 200)],
    "pink":   [(240, 150, 180), (230, 140, 170), (250, 170, 200), (240, 120, 160),
               (250, 180, 210), (230, 130, 170), (240, 160, 190), (250, 140, 180)],
    "cyan":   [(20, 180, 200), (40, 200, 220), (60, 220, 230), (80, 230, 240),
               (30, 190, 210), (50, 210, 220), (70, 225, 235), (40, 200, 215)],
    "brown":  [(100, 60, 20), (120, 80, 40), (140, 90, 50), (110, 70, 30),
               (130, 85, 45), (90, 55, 20), (105, 65, 30), (125, 75, 35)],
    "gray":   [(100, 100, 100), (120, 120, 120), (140, 140, 140), (160, 160, 160),
               (110, 110, 110), (130, 130, 130), (150, 150, 150), (170, 170, 170)],
    "black":  [(5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20),
               (25, 25, 25), (5, 10, 5), (10, 5, 10), (15, 15, 20)],
    "white":  [(250, 250, 250), (245, 245, 245), (240, 240, 240), (235, 235, 240),
               (250, 245, 245), (245, 250, 245), (240, 245, 250), (250, 250, 245)],
}

# Phrased 3 ways so the model learns the concept, not the template.
COLOR_PROMPTS = [
    ("What single color fills this image? Answer with just the color name.", "{color}."),
    ("What is the dominant color in this image?", "The dominant color is {color}."),
    ("Describe the color of this image in one word.", "{color}."),
]

SHAPES = ["circle", "square", "triangle", "star", "diamond", "heart", "cross", "line"]

SHAPE_PROMPTS = [
    ("What shape is in this image? One word.", "{shape}."),
    ("Name the geometric shape shown here.", "{shape_cap}."),
]

# Image canvas
IMG_SIZES = [(128, 128), (192, 192), (256, 256)]


def _try_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: tuple,
               cx: int, cy: int, r: int):
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
    elif shape == "star":
        import math
        points = []
        for i in range(10):
            angle = (-90 + i * 36) * math.pi / 180
            radius = r if i % 2 == 0 else r * 0.45
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=color)
    elif shape == "heart":
        import math
        points = []
        for i in range(180):
            t = i * 2 * math.pi / 180
            x = 16 * (math.sin(t) ** 3)
            y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
            points.append((cx + x * r / 17, cy + y * r / 17))
        draw.polygon(points, fill=color)
    elif shape == "cross":
        thick = r // 3
        draw.rectangle([cx - thick, cy - r, cx + thick, cy + r], fill=color)
        draw.rectangle([cx - r, cy - thick, cx + r, cy + thick], fill=color)
    elif shape == "line":
        draw.line([(cx - r, cy), (cx + r, cy)], fill=color, width=max(4, r // 8))


def _save_image(img: Image.Image, name: str) -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / name
    img.save(path)
    return str(path.relative_to(ROOT))


def emit(pairs: list, image_rel: str, instruction: str, output: str):
    pairs.append({"image": image_rel, "instruction": instruction, "output": output})


def gen_solid_colors(pairs: list, rng: random.Random):
    """12 colors × 8 shade variants × 3 prompts = 288 pairs, one image per
    (color, shade) combination. The same image gets the 3 phrased prompts."""
    total = 0
    for color_name, shades in COLORS.items():
        for i, rgb in enumerate(shades):
            size = rng.choice(IMG_SIZES)
            img = Image.new("RGB", size, color=rgb)
            rel = _save_image(img, f"solid_{color_name}_{i:02d}.png")
            for prompt_template, answer_template in COLOR_PROMPTS:
                answer = answer_template.format(color=color_name,
                                                color_cap=color_name.capitalize())
                emit(pairs, rel, prompt_template, answer)
                total += 1
    log.info("Solid colors: %d pairs (%d colors × %d shades × %d prompts)",
             total, len(COLORS), 8, len(COLOR_PROMPTS))


def gen_shapes(pairs: list, rng: random.Random):
    """Each shape in 3 different colors, asked 2 ways = 48 pairs."""
    shape_bg = (245, 245, 245)
    shape_fg_colors = [("black", (20, 20, 20)),
                       ("red",   (220, 30, 30)),
                       ("blue",  (30, 30, 220))]
    total = 0
    for shape in SHAPES:
        for color_name, rgb in shape_fg_colors:
            size = rng.choice(IMG_SIZES)
            img = Image.new("RGB", size, color=shape_bg)
            d = ImageDraw.Draw(img)
            draw_shape(d, shape, rgb, size[0] // 2, size[1] // 2, size[0] // 3)
            rel = _save_image(img, f"shape_{shape}_{color_name}.png")
            for prompt_template, answer_template in SHAPE_PROMPTS:
                answer = answer_template.format(shape=shape,
                                                shape_cap=shape.capitalize())
                emit(pairs, rel, prompt_template, answer)
                total += 1
    log.info("Shapes: %d pairs (%d shapes × %d colors × %d prompts)",
             total, len(SHAPES), 3, len(SHAPE_PROMPTS))


def gen_color_shapes(pairs: list, rng: random.Random):
    """20 random color+shape combinations × 2 prompts = 40 pairs."""
    shape_names = SHAPES[:6]  # avoid line/cross visually thin
    color_names = [c for c in COLORS.keys() if c not in ("white",)]
    combos = set()
    while len(combos) < 20:
        combos.add((rng.choice(color_names), rng.choice(shape_names)))
    total = 0
    for color_name, shape in combos:
        rgb_shade = rng.choice(COLORS[color_name])
        # Pick a contrasting background
        bg = (245, 245, 245) if color_name != "white" else (30, 30, 30)
        if color_name == "yellow":
            bg = (30, 30, 30)
        size = rng.choice(IMG_SIZES)
        img = Image.new("RGB", size, color=bg)
        d = ImageDraw.Draw(img)
        draw_shape(d, shape, rgb_shade, size[0] // 2, size[1] // 2, size[0] // 3)
        rel = _save_image(img, f"combo_{color_name}_{shape}.png")
        emit(pairs, rel, "Describe what you see in two or three words.",
             f"A {color_name} {shape}.")
        emit(pairs, rel, f"What color is the {shape}?",
             f"{color_name.capitalize()}.")
        total += 2
    log.info("Color+shape combos: %d pairs (%d combos × 2 prompts)", total, len(combos))


def gen_text_overlays(pairs: list, rng: random.Random):
    """20 single-word text images, black on white. Teaches OCR signal."""
    words = ["HELLO", "GAIA", "TEST", "CAT", "DOG", "SUN", "MOON", "YES",
             "NO", "OK", "ONE", "TWO", "RED", "BLUE", "GO", "STOP",
             "UP", "DOWN", "LEFT", "RIGHT"]
    font = _try_font(56)
    total = 0
    for word in words:
        img = Image.new("RGB", (256, 128), color=(245, 245, 245))
        d = ImageDraw.Draw(img)
        try:
            bbox = d.textbbox((0, 0), word, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except AttributeError:
            w, h = d.textsize(word, font=font)
        d.text(((256 - w) // 2, (128 - h) // 2 - 8),
               word, fill=(20, 20, 20), font=font)
        rel = _save_image(img, f"text_{word.lower()}.png")
        emit(pairs, rel, "What text is shown in this image?", word)
        total += 1
    log.info("Text overlays: %d pairs", total)


def gen_scenes(pairs: list, rng: random.Random):
    """Multi-object scenes for counting / composition reasoning."""
    scenes = [
        ("two_circles_red_blue.png",
         [("circle", COLORS["red"][2], 80, 128),
          ("circle", COLORS["blue"][2], 176, 128)],
         "How many shapes are in this image, and what colors?",
         "Two circles: one red, one blue."),
        ("three_shapes.png",
         [("circle", COLORS["red"][2], 60, 128),
          ("square", COLORS["green"][3], 128, 128),
          ("triangle", COLORS["blue"][2], 196, 128)],
         "Describe the shapes and their colors, left to right.",
         "A red circle, a green square, and a blue triangle."),
        ("star_and_heart.png",
         [("star", COLORS["yellow"][3], 90, 128),
          ("heart", COLORS["red"][2], 166, 128)],
         "What shapes do you see?",
         "A yellow star and a red heart."),
        ("four_squares_grid.png",
         [("square", COLORS["red"][2], 96, 96),
          ("square", COLORS["blue"][2], 160, 96),
          ("square", COLORS["green"][3], 96, 160),
          ("square", COLORS["yellow"][3], 160, 160)],
         "Describe what you see.",
         "Four squares: red top-left, blue top-right, green bottom-left, yellow bottom-right."),
        ("circles_line.png",
         [("circle", COLORS["purple"][3], 64, 128),
          ("circle", COLORS["cyan"][3], 128, 128),
          ("circle", COLORS["orange"][3], 192, 128)],
         "Name the colors of the circles from left to right.",
         "Purple, cyan, and orange."),
        ("two_triangles.png",
         [("triangle", COLORS["green"][3], 96, 128),
          ("triangle", COLORS["brown"][3], 160, 128)],
         "What do you see in this image?",
         "Two triangles, one green and one brown."),
        ("pink_and_gray.png",
         [("heart", COLORS["pink"][3], 96, 128),
          ("square", COLORS["gray"][3], 176, 128)],
         "Describe the shapes and colors.",
         "A pink heart and a gray square."),
        ("stars_row.png",
         [("star", COLORS["yellow"][3], 64, 128),
          ("star", COLORS["red"][2], 128, 128),
          ("star", COLORS["blue"][2], 192, 128)],
         "Count the stars and name their colors.",
         "Three stars: yellow, red, and blue."),
        ("diamond_black_white.png",
         [("diamond", COLORS["black"][2], 96, 128),
          ("diamond", COLORS["black"][2], 160, 128)],
         "What do you see?",
         "Two black diamonds on a light background."),
        ("mixed_shapes.png",
         [("circle", COLORS["red"][2], 64, 96),
          ("square", COLORS["blue"][2], 192, 96),
          ("triangle", COLORS["green"][3], 64, 160),
          ("star", COLORS["yellow"][3], 192, 160)],
         "List everything you see in this image.",
         "A red circle, a blue square, a green triangle, and a yellow star."),
    ]
    for name, items, prompt, answer in scenes:
        img = Image.new("RGB", (256, 256), color=(245, 245, 245))
        d = ImageDraw.Draw(img)
        for shape, rgb, cx, cy in items:
            draw_shape(d, shape, rgb, cx, cy, 32)
        rel = _save_image(img, name)
        emit(pairs, rel, prompt, answer)
    log.info("Scenes: %d pairs", len(scenes))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing images before generating")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.clean and IMAGES_DIR.exists():
        import shutil
        log.info("Cleaning %s...", IMAGES_DIR)
        shutil.rmtree(IMAGES_DIR)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []

    log.info("Generating expanded curriculum...")
    gen_solid_colors(pairs, rng)
    gen_shapes(pairs, rng)
    gen_color_shapes(pairs, rng)
    gen_text_overlays(pairs, rng)
    gen_scenes(pairs, rng)

    with open(OUTPUT_JSONL, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    log.info("Wrote %d vision pairs to %s", len(pairs), OUTPUT_JSONL)
    log.info("Images: %d files under %s",
             len(list(IMAGES_DIR.glob("*.png"))), IMAGES_DIR)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
