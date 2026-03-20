#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compose 3 demo screenshots into a single paper figure."""

import os

from PIL import Image

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")

imgs = [Image.open(os.path.join(FIGDIR, f"demo_tab{i}.png")) for i in range(1, 4)]

# Crop: Tab 1 keeps sidebar, Tab 2/3 crop it out
for i, img in enumerate(imgs):
    w, h = img.size
    top = 55  # remove Deploy button chrome
    if i == 0:
        # Keep sidebar for tab 1
        imgs[i] = img.crop((0, top, w - 10, h - 10))
    else:
        # Crop sidebar for tab 2/3
        imgs[i] = img.crop((290, top, w - 10, h - 10))

# Resize all to same height
target_h = min(img.size[1] for img in imgs)
for i, img in enumerate(imgs):
    w, h = img.size
    new_w = int(w * target_h / h)
    imgs[i] = img.resize((new_w, target_h), Image.LANCZOS)

# Add labels
from PIL import ImageDraw, ImageFont

labels = [
    "(a) Pipeline Demo",
    "(b) Encoding Method",
    "(c) Framework Comparison",
]
label_h = 120
labeled = []
for img, label in zip(imgs, labels):
    w, h = img.size
    new_img = Image.new("RGB", (w, h + label_h), "white")
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) / 2, h + 40), label, fill="black", font=font)
    labeled.append(new_img)

# Stack horizontally with small gap
gap = 8
total_w = sum(img.size[0] for img in labeled) + gap * (len(labeled) - 1)
max_h = max(img.size[1] for img in labeled)
composite = Image.new("RGB", (total_w, max_h), "white")

x = 0
for img in labeled:
    composite.paste(img, (x, 0))
    x += img.size[0] + gap

composite.save(os.path.join(FIGDIR, "demo.png"), dpi=(300, 300))
composite.save(os.path.join(FIGDIR, "demo.pdf"))
print(f"Saved demo.png and demo.pdf ({composite.size[0]}x{composite.size[1]})")
