# Builds the cover emblem: the pentagram with the anatomical heart at its centre.
import sys
from PIL import Image, ImageFilter, ImageChops

src = Image.open('assets/pentagram.png').convert('RGB')
# Pad with white first: the circle touches the bottom edge of the source image.
star = Image.new('RGB', (src.width + 80, src.height + 80), 'white')
star.paste(src, (40, 40))
heart = Image.open('assets/heart.webp').convert('RGBA')

# Crop the circle (dark pixels) with a small margin.
g = star.convert('L').point(lambda v: 255 if v < 128 else 0)
x0, y0, x1, y1 = g.getbbox()
pad = 20
star = star.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
W, H = star.size

# Centre of the star: mean of its five tips, measured on the source image.
tips = [(987 + 40, 165 + 40), (580 + 40, 1350 + 40), (1330 + 40, 1370 + 40), (340 + 40, 640 + 40), (1600 + 40, 640 + 40)]
cx = sum(t[0] for t in tips) / 5 - (x0 - pad)
cy = sum(t[1] for t in tips) / 5 - (y0 - pad)

heart = heart.crop(heart.getchannel('A').getbbox())
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.36
h = int(H * scale)
w = int(heart.width * h / heart.height)
heart = heart.resize((w, h), Image.LANCZOS)

# White halo so the star lines stop cleanly around the heart.
alpha = heart.getchannel('A').point(lambda v: 255 if v > 40 else 0)
halo_r = max(6, h // 28)
big = Image.new('L', (w + 4 * halo_r, h + 4 * halo_r), 0)
big.paste(alpha, (2 * halo_r, 2 * halo_r))
halo = big.filter(ImageFilter.MaxFilter(2 * halo_r + 1)).filter(ImageFilter.GaussianBlur(1.5))

ox = int(cx - w / 2)
oy = int(cy - h / 2)
out = star.convert('RGBA')
white = Image.new('RGBA', halo.size, (255, 255, 255, 255))
out.paste(white, (ox - 2 * halo_r, oy - 2 * halo_r), halo)
out.alpha_composite(heart, (ox, oy))
out.convert('RGB').save('assets/emblem.png')
print(out.size, (cx, cy), (w, h))
