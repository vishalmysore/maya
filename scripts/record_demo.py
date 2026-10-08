"""Record the browser demo for the overview video (Playwright, headless Edge) -> build/video/demo.webm

    node scripts/prepare_site.mjs --local-model build/web && python serve.py 8791   # in another shell
    python scripts/record_demo.py

A caption bar is injected into the page so the recording explains itself. Timestamps of each scene
(seconds from the start of the recording) are written to build/video/demo_scenes.json for the editor.
"""
import json
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "video"
W, H = 1280, 720

CAPTION_JS = """
(text) => {
  let c = document.getElementById('__cap');
  if (!c) {
    c = document.createElement('div'); c.id = '__cap';
    c.style.cssText = 'position:fixed;left:0;right:0;bottom:0;padding:16px 28px;background:rgba(15,15,14,.92);' +
      'color:#fff;font:600 24px/1.3 Segoe UI,system-ui,sans-serif;z-index:9999;text-align:center';
    document.body.appendChild(c);
  }
  c.textContent = text;
}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT), record_video_size={"width": W, "height": H},
                            color_scheme="light")
        pg = ctx.new_page()
        t0 = time.time()

        def mark(name):
            scenes.append({"name": name, "t": round(time.time() - t0, 2)})

        def caption(text):
            pg.evaluate(CAPTION_JS, text)

        def scroll_to(sel, offset=-12):
            pg.evaluate("([s, o]) => { const e = document.querySelector(s); window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY + o, behavior: 'smooth'}); }", [sel, offset])
            pg.wait_for_timeout(700)

        def run_preset(idx, cap_q, cap_a, hold=4200, band=(0.5, 0.5)):
            pg.select_option("#preset", str(idx))
            pg.evaluate(f"window.__maya.setBand({band[0]}, {band[1]})")
            pg.evaluate("document.getElementById('resultCard').hidden = true")
            scroll_to("#askCard")
            caption(cap_q)
            pg.wait_for_timeout(2600)
            pg.click("#runBtn")
            pg.wait_for_function("document.getElementById('status').textContent === 'Done.'", timeout=120000)
            pg.evaluate("document.getElementById('status').textContent = ''")
            scroll_to("#resultCard", -150)
            caption(cap_a)
            pg.wait_for_timeout(hold)

        pg.goto("http://localhost:8791/")
        pg.wait_for_selector("#loadBtn")
        pg.evaluate("window.__maya.load()")
        pg.wait_for_function("window.__maya.ready === true", timeout=10 * 60 * 1000)
        pg.evaluate("window.scrollTo(0, 0)")
        mark("ready")
        caption("Maya runs fully in the browser. No text leaves your device.")
        pg.wait_for_timeout(3200)

        mark("ticket")
        run_preset(2, "A support ticket and five yes/no questions", "Angry: yes.  Calm: no.  Something broken: yes.")
        mark("backup")
        run_preset(0, "An agent wants to delete a production table. A backup exists.",
                   "Cannot be undone: no.  Can be undone: yes.")
        mark("band")
        run_preset(0, "Set a band, and uncertain answers become \"not sure\"",
                   "Yes, no, or something in between.", hold=4200, band=(0.35, 0.8))
        mark("end")
        pg.wait_for_timeout(400)
        video = pg.video
        ctx.close()
        b.close()
        path = Path(video.path())
    shutil.move(str(path), OUT / "demo.webm")
    (OUT / "demo_scenes.json").write_text(json.dumps(scenes, indent=1), encoding="utf-8")
    print(scenes)


if __name__ == "__main__":
    main()
