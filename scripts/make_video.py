"""Build the short overview video -> docs/video/maya-overview.mp4 (1280x720, H.264, about one minute).

    python scripts/record_demo.py        # build/video/demo.webm + demo_scenes.json (needs the local demo server)
    python scripts/make_video.py

Scenes: title card (hero image), what Maya does, the recorded browser demo, results chart,
LLM guards vs Maya, end card with links. Cards are drawn with Pillow; ffmpeg comes from imageio-ffmpeg.
"""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "build" / "video"
OUT = ROOT / "docs" / "video"
W, H, FPS = 1280, 720, 30
BG, INK, MUTED, ACCENT, YES, NO = "#161615", "#f3f2ee", "#b4b3ac", "#7d9cf0", "#4cc38a", "#ef7a6d"
FONT_DIR = Path("C:/Windows/Fonts")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def text_block(draw, xy, text, fnt, fill, width, gap=10):
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def title_card(path):
    hero = Image.open(ROOT / "docs" / "images" / "maya-yes-no-in-between.jpg").convert("RGB")
    s = max(W / hero.width, H / hero.height)
    hero = hero.resize((round(hero.width * s), round(hero.height * s)), Image.LANCZOS)
    hero = hero.crop(((hero.width - W) // 2, (hero.height - H) // 2, (hero.width - W) // 2 + W, (hero.height - H) // 2 + H))
    img = ImageEnhance.Brightness(hero).enhance(0.8)
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for yy in range(260, H):  # soft gradient so the text sits on a dark base without a hard edge
        sd.line([(0, yy), (W, yy)], fill=(8, 8, 8, min(215, int((yy - 260) / 200 * 215))))
    img = Image.alpha_composite(img.convert("RGBA"), shade).convert("RGB")
    d = ImageDraw.Draw(img)
    d.text((70, 420), "LLM Guards vs Maya", font=font(68, True), fill=INK)
    y = text_block(d, (72, 512), "A lightweight 435M yes/no safety gate for AI agents", font(34), INK, W - 140)
    d.text((72, y + 16), "Every answer is yes, no, or something in between.", font=font(26), fill=MUTED)
    img.save(path)


def what_card(path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 60), "What Maya does", font=font(52, True), fill=INK)
    y = text_block(d, (72, 150), "Give it a text and a yes/no question. It returns the probability that the answer is yes.",
                   font(30), MUTED, W - 140)
    d.rounded_rectangle([70, y + 30, W - 70, y + 150], radius=14, fill="#22221f")
    text_block(d, (96, y + 50), "\"Delete the sessions table on production. A verified backup was taken ten minutes ago.\"",
               font(26), INK, W - 200)
    rows = [("This action can be undone if needed", "yes", "90%", YES),
            ("The action is destructive and cannot be undone", "no", "15%", NO)]
    yy = y + 200
    for q, a, p, col in rows:
        d.text((96, yy), q, font=font(30), fill=INK)
        d.text((W - 330, yy), p, font=font(30), fill=MUTED)
        d.rounded_rectangle([W - 220, yy - 4, W - 96, yy + 44], radius=24, outline=col, width=3)
        tw = d.textlength(a, font=font(28, True))
        d.text((W - 158 - tw / 2, yy + 1), a, font=font(28, True), fill=col)
        yy += 78
    d.text((72, H - 80), "No generated text. Just yes, no, and a probability.", font=font(26), fill=ACCENT)
    img.save(path)


def chart_card(path):
    img = Image.new("RGB", (W, H), "#fcfcfb")
    chart = Image.open(ROOT / "docs" / "images" / "chart-accuracy.png").convert("RGB")
    s = min((W - 80) / chart.width, (H - 150) / chart.height)
    chart = chart.resize((round(chart.width * s), round(chart.height * s)), Image.LANCZOS)
    img.paste(chart, ((W - chart.width) // 2, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 96, W, H], fill=BG)
    msg = "568 hand-labeled answers, never used for training"
    d.text(((W - d.textlength(msg, font=font(28, True))) / 2, H - 68), msg, font=font(28, True), fill=INK)
    img.save(path)


def versus_card(path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 56), "LLM guards vs Maya", font=font(52, True), fill=INK)
    cols = [("LLM guards", MUTED, ["Check a fixed list of safety categories",
                                  "Violence, hate, self-harm, prompt injection",
                                  "Often a full LLM behind the check"]),
            ("Maya", ACCENT, ["Answers any yes/no question you write",
                              "\"Can this be undone?\"  \"Is the customer angry?\"",
                              "435M encoder, runs in a browser tab"])]
    x = 70
    for name, col, items in cols:
        d.rounded_rectangle([x, 160, x + 550, 560], radius=16, fill="#22221f")
        d.text((x + 30, 184), name, font=font(38, True), fill=col)
        y = 262
        for it in items:
            d.ellipse([x + 32, y + 14, x + 44, y + 26], fill=col)
            y = text_block(d, (x + 62, y), it, font(26), INK, 460, gap=8) + 22
        x += 590
    d.text((72, H - 96), "They complement each other: a guard for content safety, Maya for your own yes/no checks.",
           font=font(24), fill=MUTED)
    img.save(path)


def end_card(path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 90), "Try Maya", font=font(64, True), fill=INK)
    rows = [("Live demo", "vishalmysore.github.io/maya"), ("Code and results", "github.com/vishalmysore/maya"),
            ("Model weights", "huggingface.co/VishalMysore/maya")]
    y = 230
    for k, v in rows:
        d.text((72, y), k, font=font(28), fill=MUTED)
        d.text((72, y + 40), v, font=font(40, True), fill=ACCENT)
        y += 120
    d.text((72, H - 100), "Open source. It still makes mistakes: use it as one signal, not the only safety check.",
           font=font(24), fill=MUTED)
    img.save(path)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    scenes = {s["name"]: s["t"] for s in json.loads((WORK / "demo_scenes.json").read_text(encoding="utf-8"))}
    demo_start, demo_len = scenes["ready"], scenes["end"] - scenes["ready"]

    cards = [("title", title_card, 5.5), ("what", what_card, 6.5), ("chart", chart_card, 6.5),
             ("versus", versus_card, 7.5), ("end", end_card, 6.0)]
    for name, fn, _ in cards:
        fn(WORK / f"card_{name}.png")
    dur = {n: t for n, _, t in cards}

    filters, labels = [], []
    order = ["title", "what", "DEMO", "chart", "versus", "end"]
    cmd = [ff, "-y", "-loglevel", "error"]
    idx = 0
    for o in order:
        if o == "DEMO":
            cmd += ["-ss", f"{demo_start:.2f}", "-t", f"{demo_len:.2f}", "-i", str(WORK / "demo.webm")]
            filters.append(f"[{idx}:v]fps={FPS},scale={W}:{H}:flags=lanczos,setsar=1,format=yuv420p,"
                           f"fade=t=in:st=0:d=0.4,fade=t=out:st={demo_len - 0.4:.2f}:d=0.4[v{idx}]")
        else:
            cmd += ["-loop", "1", "-t", str(dur[o]), "-i", str(WORK / f"card_{o}.png")]
            filters.append(f"[{idx}:v]fps={FPS},scale={W}:{H},setsar=1,format=yuv420p,"
                           f"fade=t=in:st=0:d=0.5,fade=t=out:st={dur[o] - 0.5:.2f}:d=0.5[v{idx}]")
        labels.append(f"[v{idx}]")
        idx += 1
    total = sum(dur.values()) + demo_len
    cmd += ["-f", "lavfi", "-t", f"{total:.2f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[v]")
    out = OUT / "maya-overview.mp4"
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", f"{idx}:a", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  {total:.1f} s")


if __name__ == "__main__":
    main()
