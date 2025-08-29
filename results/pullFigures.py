# export_wandb_figs.py
# Usage (saves PNGs into the current directory):
#   python export_wandb_figs.py --entity w1926284-elfbane --project LLMGeometry --run pythia410m_full --outdir .

import os
import argparse
import json
import shutil
from pathlib import Path

import wandb
from wandb import Api

# Plotly (for re-rendering charts)
import plotly.graph_objects as go
import plotly.io as pio

# Pillow (convert non-PNG images to PNG)
from PIL import Image

def slug(s: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in s)
    return safe.strip("._") or "unnamed"

def ensure_login():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit("WANDB_API_KEY not set in environment.")
    # Login for completeness (Api() will also pick it up)
    wandb.login(key=key, relogin=True, anonymous="never", force=True)

def resolve_run(api: Api, entity: str, project: str, run_slug: str):
    """Try both: direct id path and 'display_name' lookup."""
    path = f"{entity}/{project}/{run_slug}"
    try:
        return api.run(path)
    except Exception:
        # Fallback: find by display name
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_slug}, order="-created_at")
        if not runs:
            raise SystemExit(f"Run not found: {path} (also no run with display_name='{run_slug}')")
        return runs[0]

def save_plotly_png(fig_dict, out_path: Path, scale=2):
    try:
        fig = go.Figure(fig_dict)
        pio.write_image(fig, str(out_path), format="png", scale=scale)
        return True
    except Exception as e:
        print(f"[warn] Plotly PNG export failed for {out_path.name}: {e}")
        return False

def convert_to_png(src: Path, dst: Path):
    """Convert any image file to PNG using Pillow."""
    try:
        with Image.open(src) as im:
            # Handle alpha correctly
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                bg = Image.new("RGBA", im.size, (255, 255, 255, 0))
                bg.paste(im, (0, 0), im if im.mode == "RGBA" else None)
                bg.save(dst, format="PNG")
            else:
                im.convert("RGB").save(dst, format="PNG")
        return True
    except Exception as e:
        print(f"[warn] Failed to convert {src} -> {dst}: {e}")
        return False

def download_media_file(run, wandb_path: str, dest_png: Path):
    """Download a media file from W&B and ensure it's a PNG at dest_png."""
    # Download to temp name
    f = run.file(wandb_path)
    tmp_dir = dest_png.parent / ".tmp_wandb_dl"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = Path(f.download(root=str(tmp_dir), replace=True).name)

    # If already PNG, move/rename; else convert
    if local_path.suffix.lower() == ".png":
        shutil.move(str(local_path), str(dest_png))
        return True
    else:
        ok = convert_to_png(local_path, dest_png)
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        return ok

def extract_media_paths(v_dict):
    """Heuristics for W&B image-like dicts to get internal media paths."""
    paths = []
    if not isinstance(v_dict, dict):
        return paths
    # Common shapes seen in W&B history/summary for images
    # 1) {'_type': 'image-file', 'path': 'media/images/xxx.png', ...}
    # 2) {'_type': 'images/separated', 'paths': ['media/images/a.png', ...]}
    # 3) {'_type': 'image-file', 'file': {'path': 'media/images/xxx.png'}}
    for key in ("path", "paths"):
        if key in v_dict:
            if isinstance(v_dict[key], list):
                paths.extend(v_dict[key])
            elif isinstance(v_dict[key], str):
                paths.append(v_dict[key])
    if "file" in v_dict and isinstance(v_dict["file"], dict):
        if "path" in v_dict["file"]:
            paths.append(v_dict["file"]["path"])
        elif "name" in v_dict["file"]:  # older format
            paths.append(v_dict["file"]["name"])
    # Dedup & only keep media/*
    paths = [p for p in dict.fromkeys(paths) if isinstance(p, str) and p.startswith("media/")]
    return paths

def is_plotly_dict(v):
    return isinstance(v, dict) and (v.get("_type") == "plotly" or ("data" in v and "layout" in v))

def looks_like_image_dict(v):
    if not isinstance(v, dict):
        return False
    t = v.get("_type", "")
    return t.startswith("image") or t.startswith("images") or t in {"image-file"}

def export_run(entity: str, project: str, run_slug: str, outdir: Path):
    ensure_login()
    api = wandb.Api(timeout=60)
    run = resolve_run(api, entity, project, run_slug)
    run_name = run.name or run.id
    run_tag = slug(run_name)

    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Exporting figures for: {run.entity}/{run.project}/{run.id}  (name: {run_name})")
    print(f"[info] Saving PNGs to: {outdir}")

    saved = 0
    seen = set()

    def save_plotly_from_value(key, step, v):
        nonlocal saved
        if "plotly" in v:
            fig_dict = v["plotly"]
        else:
            fig_dict = v  # already looks like {'data': ..., 'layout': ...}
        fname = f"{run_tag}__step{step:06d}__{slug(key)}.png"
        dest = outdir / fname
        if (key, step, "plotly") in seen:
            return
        if save_plotly_png(fig_dict, dest):
            seen.add((key, step, "plotly"))
            saved += 1

    def save_images_from_value(key, step, v):
        nonlocal saved
        paths = extract_media_paths(v)
        if not paths:
            return
        for idx, p in enumerate(paths):
            fname = f"{run_tag}__step{step:06d}__{slug(key)}"
            if len(paths) > 1:
                fname += f"_{idx:02d}"
            fname += ".png"
            dest = outdir / fname
            sig = (key, step, p)
            if sig in seen:
                continue
            try:
                ok = download_media_file(run, p, dest)
                if ok:
                    seen.add(sig)
                    saved += 1
            except Exception as e:
                print(f"[warn] Failed downloading {p}: {e}")

    # 1) Scan history (streaming, memory-safe)
    for row in run.scan_history(page_size=2000):
        step = row.get("_step", -1)
        for k, v in row.items():
            if k.startswith("_"):
                continue
            if is_plotly_dict(v):
                save_plotly_from_value(k, step, v)
            elif looks_like_image_dict(v):
                save_images_from_value(k, step, v)

    # 2) Also look into run.summary for final artifacts
    for k, v in dict(run.summary).items():
        step = run.lastHistoryStep or -1
        if is_plotly_dict(v):
            save_plotly_from_value(k, step, v)
        elif looks_like_image_dict(v):
            save_images_from_value(k, step, v)

    print(f"[done] Saved {saved} PNG file(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", required=True, help="Run *id* or *name* (script tries both)")
    parser.add_argument("--outdir", default=".", help="Where to save PNGs (default: current folder)")
    args = parser.parse_args()
    export_run(args.entity, args.project, args.run, Path(args.outdir))
