"""Kirby-style icon generator for Windows Rainmeter.

Draws a soft pink puffball (Kirby) and exports PNGs at several sizes
plus a multi-resolution .ico. No external assets needed.

Eye size is tunable so the face can be made more/less cute.
"""
import os, sys
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
SS = 4            # supersampling factor
BASE = 512       # logical canvas
S = BASE * SS    # work canvas

# palette ---------------------------------------------------------------
BODY      = (244, 167, 193)   # soft pink body
BODY_DARK = (228, 138, 170)   # bottom shading / outline
BODY_LITE = (252, 214, 226)   # top-left highlight
FOOT      = (220, 58, 78)     # red feet
FOOT_DARK = (176, 38, 58)
EYE_NAVY  = (28, 30, 80)
EYE_BLUE  = (70, 110, 224)
WHITE     = (255, 255, 255)
CHEEK     = (240, 110, 142)
MOUTH     = (120, 36, 52)


def E(d, cx, cy, rx, ry, fill, outline=None, width=0):
    d.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=fill, outline=outline, width=width)


def make_foot(w, h, angle):
    layer = Image.new("RGBA", (w*2, h*2), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    dl.ellipse([w*0.5, h*0.5, w*1.5, h*1.5], fill=FOOT,
               outline=FOOT_DARK, width=int(6*SS))
    return layer.rotate(angle, expand=True, resample=Image.BICUBIC)


def make_arm(w, h, angle):
    layer = Image.new("RGBA", (w*2, h*2), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    dl.ellipse([w*0.5, h*0.5, w*1.5, h*1.5], fill=BODY,
               outline=BODY_DARK, width=int(5*SS))
    return layer.rotate(angle, expand=True, resample=Image.BICUBIC)


def make_eye(ew, eh):
    eye = Image.new("RGBA", (ew, eh), (0, 0, 0, 0))
    de = ImageDraw.Draw(eye)
    m = Image.new("L", (ew, eh), 0)
    ImageDraw.Draw(m).ellipse([0, 0, ew-1, eh-1], fill=255)
    de.rectangle([0, 0, ew, eh], fill=EYE_NAVY)
    de.ellipse([int(ew*0.02), int(eh*0.32), int(ew*0.98), int(eh*1.25)], fill=EYE_BLUE)
    de.ellipse([int(ew*0.24), int(eh*0.08), int(ew*0.78), int(eh*0.40)], fill=WHITE)
    de.ellipse([int(ew*0.30), int(eh*0.78), int(ew*0.70), int(eh*0.95)], fill=WHITE)
    eye.putalpha(m)
    return eye


def draw_kirby(eye_w_f=0.30, eye_h_f=0.50):
    """Render Kirby at BASE resolution. eye_*_f scale the eyes vs body R."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    cx, cy = S//2, int(S*0.52)
    R = int(S*0.34)

    # feet
    lf = make_foot(int(R*0.62), int(R*0.42), 18)
    img.alpha_composite(lf, (int(cx - R*0.62 - lf.width/2), int(cy + R*0.55 - lf.height/2)))
    rf = make_foot(int(R*0.66), int(R*0.46), -28)
    img.alpha_composite(rf, (int(cx + R*0.42 - rf.width/2), int(cy + R*0.60 - rf.height/2)))

    # arms
    aw, ah = int(R*0.34), int(R*0.24)
    la = make_arm(aw, ah, -35)
    img.alpha_composite(la, (int(cx - R*0.92 - la.width/2), int(cy - R*0.05 - la.height/2)))
    ra = make_arm(aw, ah, 35)
    img.alpha_composite(ra, (int(cx + R*0.92 - ra.width/2), int(cy - R*0.30 - ra.height/2)))

    # body
    d = ImageDraw.Draw(img)
    E(d, cx, cy, R, R, BODY, outline=BODY_DARK, width=int(5*SS))

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([cx-R, cy-R, cx+R, cy+R], fill=255)

    shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    E(ImageDraw.Draw(shade), cx, int(cy + R*0.38), int(R*0.92), int(R*0.62), (*BODY_DARK, 120))
    shade = shade.filter(ImageFilter.GaussianBlur(int(18*SS)))
    img.paste(shade, (0, 0), Image.composite(shade.split()[3], Image.new("L", (S, S), 0), mask))

    hi = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    E(ImageDraw.Draw(hi), int(cx - R*0.36), int(cy - R*0.44), int(R*0.34), int(R*0.24), (*BODY_LITE, 120))
    hi = hi.filter(ImageFilter.GaussianBlur(int(26*SS)))
    img.paste(hi, (0, 0), Image.composite(hi.split()[3], Image.new("L", (S, S), 0), mask))

    # eyes
    ew, eh = int(R*eye_w_f), int(R*eye_h_f)
    ex = int(R*0.26)
    ey = int(cy - R*0.14)
    img.alpha_composite(make_eye(ew, eh), (int(cx - ex - ew/2), int(ey - eh/2)))
    img.alpha_composite(make_eye(ew, eh), (int(cx + ex - ew/2), int(ey - eh/2)))

    # cheeks
    ch = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dc = ImageDraw.Draw(ch)
    E(dc, int(cx - R*0.50), int(cy + R*0.12), int(R*0.13), int(R*0.085), (*CHEEK, 235))
    E(dc, int(cx + R*0.50), int(cy + R*0.12), int(R*0.13), int(R*0.085), (*CHEEK, 235))
    ch = ch.filter(ImageFilter.GaussianBlur(int(6*SS)))
    img.alpha_composite(ch)

    # mouth
    d = ImageDraw.Draw(img)
    mx, my = cx, int(cy + R*0.15)
    mw = int(R*0.10)
    d.arc([mx-mw, my-int(mw*0.6), mx+mw, my+int(mw*1.4)], start=20, end=160,
          fill=MOUTH, width=int(5*SS))
    d.line([mx-int(mw*0.55), my, mx, my+int(mw*0.5)], fill=MOUTH, width=int(5*SS))
    d.line([mx+int(mw*0.55), my, mx, my+int(mw*0.5)], fill=MOUTH, width=int(5*SS))

    return img.resize((BASE, BASE), Image.LANCZOS)


def export(final, prefix="kirby"):
    png_master = os.path.join(OUT, f"{prefix}.png")
    final.save(png_master)
    sizes = [256, 128, 64, 48, 32, 16]
    for s in sizes:
        final.resize((s, s), Image.LANCZOS).save(os.path.join(OUT, f"{prefix}_{s}.png"))
    final.save(os.path.join(OUT, f"{prefix}.ico"), sizes=[(s, s) for s in sizes])
    print("wrote:", png_master, "+ sizes", sizes, "+ .ico")


# preset eye sizes (w_factor, h_factor) --------------------------------
PRESETS = {
    "big":    (0.40, 0.66),   # original (too big / scary)
    "medium": (0.30, 0.50),
    "small":  (0.22, 0.38),
    "tiny":   (0.17, 0.30),
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # side-by-side comparison strip of all presets
        order = ["big", "medium", "small", "tiny"]
        pad = 24
        tile = 256
        strip = Image.new("RGBA", (tile*len(order) + pad*(len(order)+1), tile + 60),
                          (250, 250, 250, 255))
        dd = ImageDraw.Draw(strip)
        for i, name in enumerate(order):
            wf, hf = PRESETS[name]
            k = draw_kirby(wf, hf).resize((tile, tile), Image.LANCZOS)
            x = pad + i*(tile+pad)
            strip.alpha_composite(k, (x, pad))
            dd.text((x + tile//2 - 20, tile + pad + 8), name, fill=(40, 40, 40, 255))
        strip.convert("RGB").save(os.path.join(OUT, "compare.png"))
        print("wrote compare.png")
    else:
        preset = sys.argv[1] if len(sys.argv) > 1 else "small"
        wf, hf = PRESETS[preset]
        export(draw_kirby(wf, hf))
        print("preset:", preset, PRESETS[preset])
