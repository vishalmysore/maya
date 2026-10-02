"""Capture documentation screenshots of the browser demo (headless Edge via Playwright).

    node scripts/prepare_site.mjs --local-model build/web && python serve.py 8791   # in another shell
    python scripts/screenshots.py --url http://localhost:8791/

Also re-runs the tokenizer / probability parity check in the browser (dist/parity_cases.json if present).
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = [  # (file, preset index, (t_no, t_yes))
    ("demo-guardrail-backup.png", 0, (0.5, 0.5)),
    ("demo-guardrail-not-sure.png", 0, (0.3, 0.8)),
    ("demo-read-only-query.png", 1, (0.5, 0.5)),
    ("demo-support-ticket-miss.png", 2, (0.5, 0.5)),
    ("demo-code-change.png", 3, (0.5, 0.5)),
    ("demo-it-incident.png", 4, (0.5, 0.5)),
    ("demo-product-review.png", 5, (0.5, 0.5)),
    ("demo-unseen-library-notice.png", 6, (0.5, 0.5)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8791/")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1100, "height": 1000}, device_scale_factor=2, color_scheme="light")
        pg.goto(args.url)
        pg.wait_for_selector("#loadBtn")
        pg.screenshot(path=str(out / "demo-start.png"))
        pg.evaluate("window.__maya.load()")
        pg.wait_for_function("window.__maya.ready === true", timeout=10 * 60 * 1000)
        print("loaded:", pg.inner_text("#status"))
        pg.locator("#loadCard").screenshot(path=str(out / "demo-model-loaded.png"))
        if (ROOT / "dist" / "parity_cases.json").exists():
            res = pg.evaluate("""async () => {
              const cases = await (await fetch('./parity_cases.json')).json();
              let idMismatch = 0, maxDp = 0, flips = 0;
              for (const c of cases) {
                const ids = window.__maya.encode(c.text, c.statement);
                if (JSON.stringify(ids) !== JSON.stringify(c.ids)) idMismatch++;
                const r = (await window.__maya.ask(c.text, [c.statement], true)).results[0];
                maxDp = Math.max(maxDp, Math.abs(r.p_yes - c.p_torch));
                if ((r.p_yes >= 0.5) !== (c.p_torch >= 0.5)) flips++;
              }
              return {n: cases.length, idMismatch, maxDp, flips};
            }""")
            print("browser parity vs Python/PyTorch:", res)
            (ROOT / "results" / "browser_parity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        results = {}
        for fname, idx, (t_no, t_yes) in SHOTS:
            pg.select_option("#preset", str(idx))
            pg.evaluate(f"window.__maya.setBand({t_no}, {t_yes})")
            pg.evaluate("window.__maya.run()")
            pg.wait_for_function("document.getElementById('status').textContent === 'Done.'", timeout=120000)
            pg.evaluate("document.getElementById('askCard').scrollIntoView()")
            pg.locator("#askCard").screenshot(path=str(out / fname.replace("demo-", "input-")))
            pg.locator("#resultCard").screenshot(path=str(out / fname))
            results[fname] = pg.evaluate("window.__maya.last.map(r => [r.statement, +r.p_yes.toFixed(3), r.answer])")
            print(fname, results[fname])
            pg.evaluate("document.getElementById('status').textContent = ''")
        pg.evaluate("window.__maya.setBand(0.5, 0.5)")
        pg.select_option("#preset", "2")
        pg.evaluate("window.__maya.run()")
        pg.wait_for_function("document.getElementById('status').textContent === 'Done.'", timeout=120000)
        pg.screenshot(path=str(out / "demo-full-page.png"), full_page=True)
        (ROOT / "results" / "demo_answers.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        b.close()


if __name__ == "__main__":
    main()
