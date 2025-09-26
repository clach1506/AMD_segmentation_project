# interface_tk.py — Minimal Tkinter UI for automatic segmentation
# Run: python interface_tk.py  (inside the environment where torch + your utils are installed)

import os
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# NEW: matplotlib backend for Tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image

# ---- Project imports (adapt path if needed)
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

import torch
from models.rnn import FlexibleGRU
from utils import load_model, segment_series, segment_series_to_colormap,segment_series_distance

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def choose_dir(entry_widget):
    d = filedialog.askdirectory()
    if d:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, d)

def choose_file(entry_widget, title="Select file",
                filetypes=(("Image files", "*.png *.tif *.tiff"), ("All files", "*.*"))):
    from tkinter import filedialog
    f = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=root)
    if f:
        entry_widget.delete(0, "end")
        entry_widget.insert(0, f)

def append_log(text):
    log_box.configure(state="normal")
    log_box.insert(tk.END, text + "\n")
    log_box.see(tk.END)
    log_box.configure(state="disabled")

def _list_pngs(folder: str, prefix: str | None = None):
    p = Path(folder)
    if not p.exists():
        return []
    cands = sorted([q for q in p.glob("*.png")])
    if prefix:
        cands = [q for q in cands if q.name.startswith(prefix)] or cands
    return cands

def _open_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)

# -------------------------------------------------
# Preview logic (matplotlib in Tk)
# -------------------------------------------------
last_outdir_masks = None
last_outdir_cmaps = None

def show_array_on_ax(ax, arr, cmap=None, vmin=None, vmax=None):
    ax.clear()
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    canvas.draw_idle()

def update_preview(*_):
    mode = cb_preview.get()

    if mode == "Onset colormap":
        if not last_outdir_cmaps:
            append_log("No colormap folder yet. Run segmentation first.")
            return
        imgs = _list_pngs(last_outdir_cmaps)
        if not imgs:
            append_log("No colormap PNG found.")
            return
        arr = np.asarray(Image.open(imgs[0]).convert("RGB"))
        show_array_on_ax(ax, arr)  # already colored PNG

    elif mode == "Total growth (last - first)":
        if not last_outdir_masks:
            append_log("No masks folder yet. Run segmentation first.")
            return
        segs = _list_pngs(last_outdir_masks, prefix="seg_")
        if len(segs) < 2:
            append_log("Need at least two seg_* PNGs to compute growth.")
            return
        first = _open_gray(segs[0]) > 0
        last  = _open_gray(segs[-1]) > 0
        growth = (last & (~first)).astype(np.uint8)  # 1 where new GA appeared
        # Display with a perceptual colormap; background 0, growth 1
        show_array_on_ax(ax, growth, cmap="magma", vmin=0, vmax=1)

def _is_image_file(p: str) -> bool:
    import os
    return os.path.splitext(p)[1].lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

# -------------------------------------------------
# Segmentation runner (thread)
# -------------------------------------------------
def run_segmentation():
    global last_outdir_masks, last_outdir_cmaps
    btn_run.configure(state="disabled")
    progress.start(10)
    append_log("Starting segmentation...")

    try:
        # Read UI values
        input_folder = ent_input.get().strip()
        output_root = ent_output.get().strip()
        model_type = cb_model.get()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        T = int(spin_T.get())
        threshold = float(ent_thr.get())
        min_duration = int(spin_min_duration.get())
        min_size = int(spin_min_size.get())
        sigma = float(ent_sigma.get())
        smooth_temporal = bool(var_smooth.get())
        batch_size = int(spin_batch.get())
        ignore_first_frame_onset = bool(var_ignore_first.get())

        pixel_model_path = ent_pixel.get().strip()
        patch_model_path = ent_patch.get().strip()
        distance_model_path = ent_distance.get().strip()
        ref_mask_path = ent_refmask.get().strip()

        if not input_folder or not os.path.isdir(input_folder):
            raise ValueError("Please select a valid input images folder.")

        Path(output_root).mkdir(parents=True, exist_ok=True)

        # Load selected model
        append_log(f"Device: {device}")
        if model_type == "pixel":
            model = load_model(pixel_model_path, model_class=FlexibleGRU, device=device,
                               input_size=1, hidden_size=64, num_layers=2)
            seg_folder_name = "seg_pixel_model"
        elif model_type == "patch":
            model = load_model(patch_model_path, model_class=FlexibleGRU, device=device,
                               input_size=9, hidden_size=64, num_layers=2)
            seg_folder_name = "seg_patch_model"
        else:
            model = load_model(distance_model_path, model_class=FlexibleGRU, device=device,
                               input_size=2, hidden_size=64, num_layers=2)
            seg_folder_name = "seg_distance_model"

        append_log(f"Model '{model_type}' loaded.")

        # 1) Binary masks


        if model_type == "distance":
            if not ref_mask_path:
                raise ValueError("Please choose a reference GA mask file (.png/.tif).")
            if os.path.isdir(ref_mask_path) or not os.path.isfile(ref_mask_path) or not _is_image_file(ref_mask_path):
                raise ValueError(f"Invalid mask file: {ref_mask_path}")

            outdir_masks = segment_series_distance(
                model=model,
                T=T,
                device=device,
                threshold=threshold,
                min_duration=min_duration,
                min_size=min_size,
                sigma=sigma,
                smooth_temporal=smooth_temporal,
                batch_size=batch_size,
                seg_prefix="seg_",
                folder_name="seg_distance_model",
                seg_mode="L",
                input_folder=input_folder,
                output_dir=output_root,
                ref_mask_path=ref_mask_path,   # <-- récupéré de l’UI
            )
        else:
            outdir_masks = segment_series(
                model=model,
                model_type=model_type,
                T=T,
                device=device,
                threshold=threshold,
                min_duration=min_duration,
                min_size=min_size,
                sigma=sigma,
                smooth_temporal=smooth_temporal,
                batch_size=batch_size,
                seg_prefix="seg_",
                folder_name="seg_patch_model" if model_type=="patch" else "seg_pixel_model",
                seg_mode="L",
                input_folder=input_folder,
                output_dir=output_root
            )

            append_log(f"Binary masks saved in: {outdir_masks}")

        # 2) Onset colormaps
        outdir_cmaps = segment_series_to_colormap(
            model=model,
            model_type=model_type,
            T=T,
            device=device,
            threshold=threshold,
            min_duration=min_duration,
            min_size=min_size,
            sigma=sigma,
            smooth_temporal=smooth_temporal,
            batch_size=batch_size,
            folder_name=seg_folder_name + "_colormaps",
            stride = 8,
            cmap_name = "turbo",
            absolute_index= False,
            save_indices_tiff = True,
            input_folder=input_folder,
            output_dir=output_root
        )
        append_log(f"Onset colormaps saved in: {outdir_cmaps}")

        # Store for preview and refresh
        last_outdir_masks = outdir_masks
        last_outdir_cmaps = outdir_cmaps
        update_preview()

        append_log("Done.")
        messagebox.showinfo("Segmentation", "Processing finished successfully.")

    except Exception as e:
        append_log(f"ERROR: {e}")
        messagebox.showerror("Segmentation error", str(e))
    finally:
        progress.stop()
        btn_run.configure(state="normal")

def on_click_run():
    threading.Thread(target=run_segmentation, daemon=True).start()

# -------------------------------------------------
# UI layout
# -------------------------------------------------
root = tk.Tk()
root.title("Automatic Segmentation — Minimal UI")
root.geometry("980x720")

padx = 8
pady = 6

# Paths
frm_paths = ttk.LabelFrame(root, text="Paths")
frm_paths.pack(fill="x", padx=padx, pady=pady)

ttk.Label(frm_paths, text="Images folder:").grid(row=0, column=0, sticky="w", padx=5, pady=4)
ent_input = ttk.Entry(frm_paths, width=80)
ent_input.grid(row=0, column=1, sticky="we", padx=5, pady=4)
ttk.Button(frm_paths, text="Browse…", command=lambda: choose_dir(ent_input)).grid(row=0, column=2, padx=5, pady=4)

ttk.Label(frm_paths, text="Output folder:").grid(row=1, column=0, sticky="w", padx=5, pady=4)
ent_output = ttk.Entry(frm_paths, width=80)
ent_output.grid(row=1, column=1, sticky="we", padx=5, pady=4)
ttk.Button(frm_paths, text="Browse…", command=lambda: choose_dir(ent_output)).grid(row=1, column=2, padx=5, pady=4)

# Models
frm_models = ttk.LabelFrame(root, text="Models")
frm_models.pack(fill="x", padx=padx, pady=pady)

ttk.Label(frm_models, text="Model type:").grid(row=0, column=0, sticky="w", padx=5, pady=4)
cb_model = ttk.Combobox(frm_models, values=["pixel", "patch", "distance"], state="readonly")
cb_model.current(1)
cb_model.grid(row=0, column=1, sticky="w", padx=5, pady=4)

ttk.Label(frm_models, text="Pixel model (.pt):").grid(row=1, column=0, sticky="w", padx=5, pady=4)
ent_pixel = ttk.Entry(frm_models, width=60); ent_pixel.insert(0, "models/pixel_model.pt")
ent_pixel.grid(row=1, column=1, sticky="we", padx=5, pady=4)
ttk.Button(frm_models, text="Browse…", command=lambda: choose_file(ent_pixel)).grid(row=1, column=2, padx=5, pady=4)

ttk.Label(frm_models, text="Patch model (.pt):").grid(row=2, column=0, sticky="w", padx=5, pady=4)
ent_patch = ttk.Entry(frm_models, width=60); ent_patch.insert(0, "models/patch_model.pt")
ent_patch.grid(row=2, column=1, sticky="we", padx=5, pady=4)
ttk.Button(frm_models, text="Browse…", command=lambda: choose_file(ent_patch)).grid(row=2, column=2, padx=5, pady=4)

ttk.Label(frm_models, text="Distance model (.pt):").grid(row=3, column=0, sticky="w", padx=5, pady=4)
ent_distance = ttk.Entry(frm_models, width=60); ent_distance.insert(0, "models/distance_model.pt")
ent_distance.grid(row=3, column=1, sticky="we", padx=5, pady=4)
ttk.Button(frm_models, text="Browse…", command=lambda: choose_file(ent_distance)).grid(row=3, column=2, padx=5, pady=4)

ttk.Label(frm_models, text="Ref. mask (for 'distance')").grid(row=4, column=0, sticky="w", padx=5, pady=4)
ent_refmask = ttk.Entry(frm_models, width=60)
ent_refmask.grid(row=4, column=1, sticky="we", padx=5, pady=4)
ttk.Button(
    frm_models, text="Browse…",
    command=lambda: choose_file(
        ent_refmask,
        title="Reference mask",
        filetypes=(("Image files", "*.png *.tif *.tiff"), ("All files", "*.*"))
    )
).grid(row=4, column=2, padx=5, pady=4)

# Parameters
frm_params = ttk.LabelFrame(root, text="Parameters")
frm_params.pack(fill="x", padx=padx, pady=pady)

ttk.Label(frm_params, text="T:").grid(row=0, column=0, sticky="w", padx=5, pady=4)
spin_T = ttk.Spinbox(frm_params, from_=1, to=512, width=6); spin_T.set(16); spin_T.grid(row=0, column=1, padx=5, pady=4)

ttk.Label(frm_params, text="Threshold:").grid(row=0, column=2, sticky="w", padx=5, pady=4)
ent_thr = ttk.Entry(frm_params, width=6); ent_thr.insert(0, "0.5"); ent_thr.grid(row=0, column=3, padx=5, pady=4)

ttk.Label(frm_params, text="Min duration:").grid(row=0, column=4, sticky="w", padx=5, pady=4)
spin_min_duration = ttk.Spinbox(frm_params, from_=1, to=50, width=6); spin_min_duration.set(2); spin_min_duration.grid(row=0, column=5, padx=5, pady=4)

ttk.Label(frm_params, text="Min size:").grid(row=0, column=6, sticky="w", padx=5, pady=4)
spin_min_size = ttk.Spinbox(frm_params, from_=0, to=100000, width=8); spin_min_size.set(20); spin_min_size.grid(row=0, column=7, padx=5, pady=4)

ttk.Label(frm_params, text="Sigma:").grid(row=1, column=0, sticky="w", padx=5, pady=4)
ent_sigma = ttk.Entry(frm_params, width=6); ent_sigma.insert(0, "0.7"); ent_sigma.grid(row=1, column=1, padx=5, pady=4)

ttk.Label(frm_params, text="Batch size:").grid(row=1, column=2, sticky="w", padx=5, pady=4)
spin_batch = ttk.Spinbox(frm_params, from_=1, to=100000, width=8); spin_batch.set(1024); spin_batch.grid(row=1, column=3, padx=5, pady=4)

var_smooth = tk.BooleanVar(value=False)
ttk.Checkbutton(frm_params, text="Temporal smoothing", variable=var_smooth).grid(row=1, column=4, padx=5, pady=4)

var_ignore_first = tk.BooleanVar(value=False)
ttk.Checkbutton(frm_params, text="Ignore onset at frame 1", variable=var_ignore_first).grid(row=1, column=5, padx=5, pady=4)

# Run + progress
frm_run = ttk.Frame(root)
frm_run.pack(fill="x", padx=padx, pady=pady)
btn_run = ttk.Button(frm_run, text="Run segmentation", command=on_click_run)
btn_run.pack(side="left")
progress = ttk.Progressbar(frm_run, mode="indeterminate")
progress.pack(fill="x", expand=True, padx=10)

# NEW: Preview panel
frm_prev = ttk.LabelFrame(root, text="Preview")
frm_prev.pack(fill="both", expand=True, padx=padx, pady=pady)

cb_preview = ttk.Combobox(frm_prev, values=["Onset colormap", "Total growth (last - first)"], state="readonly", width=30)
cb_preview.current(0)
cb_preview.grid(row=0, column=0, sticky="w", padx=5, pady=4)
cb_preview.bind("<<ComboboxSelected>>", update_preview)

fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frm_prev)
canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
frm_prev.columnconfigure(0, weight=1)
frm_prev.rowconfigure(1, weight=1)

# Log box
frm_log = ttk.LabelFrame(root, text="Log")
frm_log.pack(fill="both", expand=True, padx=padx, pady=pady)
log_box = tk.Text(frm_log, height=10, state="disabled")
log_box.pack(fill="both", expand=True)

# Defaults
ent_input.insert(0, str(ROOT.parent / "data" / "series_example"))
ent_output.insert(0, str(ROOT.parent / "outputs"))

root.mainloop()
