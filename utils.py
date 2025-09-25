
import torch, numpy as np, torch.functional as F, tkinter as tk
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm


from tkinter import filedialog

from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter,distance_transform_edt
from skimage.morphology import remove_small_objects

##  Loading functions

def load_ir_images(series_input):
    """
    Load IR images as a stack of size (T, H, W).

    Args:
        series_input: either
            - a folder path (str or Path) containing .png images
            - a list of image file paths (list of str or Path)

    Returns:
        np.ndarray of shape (T, H, W)
    """
    if isinstance(series_input, (str, Path)):  
        # Input is a folder → collect .png files
        series_path = Path(series_input)
        image_files = sorted(series_path.glob("*.png"))
    elif isinstance(series_input, (list, tuple)):
        # Input is already a list of files
        image_files = sorted(map(Path, series_input))
    else:
        raise TypeError("series_input must be a folder path or a list of image paths")

    if not image_files:
        raise FileNotFoundError(f"No images found in {series_input}")

    images = []
    for path in image_files:
        try:
            with Image.open(path) as img:
                images.append(np.array(img))
        except Exception as e:
            print(f"Error with {path.name}: {e}")
            raise e

    return np.stack(images)  # (T, H, W)

def load_model(path, model_class, device="cpu", **model_kwargs):
    """
    Load a trained model from checkpoint.
    Parameters:
    - path: path to the saved model (.pt file).
    - model_class: class of the model to instantiate (e.g., PixelGRU, GRU2).
    - device: 'cpu' or 'cuda'.
    - model_kwargs: optional arguments to pass when instantiating the model.
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

##  Processing helpers functions

def compute_validity_mask(ir_series, threshold=0):
    """
    Calcule un masque binaire (H, W) des pixels valides sur toute la série.
    Un pixel est valide s’il dépasse le seuil d’intensité sur au moins UNE frame.

    Args:
        ir_series (np.ndarray): Série IR de forme (T, H, W)
        threshold : Seuil d’intensité fixé à 0 généralement
    Returns:
        np.ndarray: Masque de validité (H, W), bool
    """
    valid = ir_series > threshold  # (T, H, W)
    validity_mask = np.any(valid, axis=0)  # (H, W)
    return validity_mask

def predict_on_subseries(
    model,
    X_np,
    device="cpu",
    smooth=False,
    batch_size=4096,
    return_proba=True,
    threshold=0.5,
):
    """
    Forward modèle sur des sous-séries et renvoie par défaut des probabilités.

    Args:
        model: PyTorch module. Sortie attendue: (N, T) ou (N, T, 1) en logits.
        X_np: np.ndarray (N, T, C) — C=1 (pixel) ou C=P (patch).
        device: "cpu" ou "cuda".
        smooth: lissage temporel 1D sur les probas (fenêtre 3).
        batch_size: taille de lot pour réduire l'empreinte mémoire.
        return_proba: si True -> renvoie float32 dans [0,1], sinon binaire.
        threshold: seuil si return_proba=False.

    Returns:
        np.ndarray (N, T) float32 si return_proba=True, sinon uint8 binaire.
    """
    assert X_np.ndim == 3, f"X_np must be (N, T, C), got {X_np.shape}"
    N, T, C = X_np.shape

    model = model.to(device)
    model.eval()

    probs_all = np.empty((N, T), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = torch.from_numpy(X_np[start:end]).float().to(device)  # (B, T, C)

            logits = model(x)  # (B, T) ou (B, T, 1)
            if logits.ndim == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

            probs = torch.sigmoid(logits)  # (B, T)

            if smooth:
                # lissage conv 1D sur T
                p = probs.unsqueeze(1)             # (B, 1, T)
                kernel = torch.tensor([[[1/3, 1/3, 1/3]]], device=probs.device)
                p = F.conv1d(p, kernel, padding=1) # (B, 1, T)
                probs = p.squeeze(1)                # (B, T)

            probs_all[start:end] = probs.float().cpu().numpy()

    if return_proba:
        return probs_all.astype(np.float32)
    else:
        return (probs_all >= float(threshold)).astype(np.uint8)

def enforce_min_duration(y_bin: np.ndarray, min_duration: int) -> np.ndarray:
    """
    Vectorized min-duration filter. y_bin: (N, T) bool/uint8 -> returns bool.
    """
    y = (y_bin > 0)
    if not min_duration or min_duration <= 1:
        return y

    N, T = y.shape
    # Trouver débuts/fins de runs via dérivée discrète
    pad = np.pad(y, ((0,0),(1,1)), constant_values=False)
    starts = (~pad[:, :-1] & pad[:, 1:])   # False->True
    ends   = (pad[:, :-1] & ~pad[:, 1:])   # True->False

    # Indices absolus
    start_idx = np.where(starts)[1]
    end_idx   = np.where(ends)[1]
    row_idx_s = np.where(starts)[0]
    row_idx_e = np.where(ends)[0]

    # Par sécurité, ils sont appariés dans l'ordre
    # (pad garantit un nombre égal de starts/ends par ligne)
    out = np.zeros_like(y, dtype=bool)
    for n in range(N):
        s = start_idx[row_idx_s == n]
        e = end_idx[row_idx_e == n]
        # Conserver uniquement les runs assez longs
        keep = (e - s) >= min_duration
        for ss, ee in zip(s[keep], e[keep]):
            out[n, ss:ee] = True
    return out

## Core functions : segmentations of image series

def segment_series(
    model,                             # modèle PyTorch (pixel OU patch)
    model_type: str = "pixel",         # "pixel" ou "patch"
    T: int = 16,
    device: str = "cpu",
    threshold: float = 0.5,            # seuil de binarisation
    min_duration: int = 3,             # 0 pour désactiver
    min_size: int = 20,                # 0 pour désactiver
    sigma: float = 0.0,                # 0 pour désactiver (lissage spatial des probas)
    smooth_temporal: bool = False,     # lissage temporel dans predict_on_subseries
    batch_size: int = 1024,
    seg_prefix: str = "seg_",
    folder_name : str = "SEG_",
    seg_mode: str = "L",               # "L" (0/255) ou "1" (1-bit)
):
    """
    This function opens a windows to select a folder of images IR to segment, the images must be already normalized to ensure coherent results.
    This functions uses a pixel-pretrained model or a patch-pretrained model as wanted. 
    Several parameters can be used : 
    - Model type : ie pixel or patch (patch usually offers better results)
    - Threshold : threshold used to binarize prediction of the model 
    - Min Duration : keep only runs of 1s with length >= min_duration, per pixel row
    - Min size : keep only regions of pixels with size >= min_size
    - sigma : Gaussian blur
    - Smooth_temporal : (bool) used to smooth predictions 
    - Batch size
    - Seg_prefix : name of the segmented images + index 
    """
    assert model_type in ("pixel", "patch"), "model_type must be 'pixel' or 'patch'"

    # ---- 1) Folder choice and loading of the images serie ----
    root = tk.Tk(); root.withdraw()
    image_dir = filedialog.askdirectory(title="Select images folder")
    if not image_dir:
        raise RuntimeError("No selected folder")
    
    series = load_ir_images(image_dir)  # <-- ta fonction
    if series.ndim != 3:
        raise ValueError(f"load_ir_images doit renvoyer (T,H,W), obtenu {series.shape}")

    T_orig, H, W = series.shape

    # ---- 2) Padding of short series or division in subseries ----
    if T_orig < T:
        pad = T - T_orig
        series_pad = np.concatenate([series, np.zeros((pad, H, W), dtype=series.dtype)], axis=0)
    else:
        series_pad = series[:T]
    T_eff = series_pad.shape[0]

    # ---- 3) Validity mask to keep only relevant regions ----
    mask_valid = compute_validity_mask(series_pad, threshold=0)  # (H, W) bool
    vi, vj = np.where(mask_valid)

    # ---- 4) Eliminate borders when patch model is activated ----
    if model_type == "patch":
        patch_size = 3
        off = patch_size // 2
        keep = (vi >= off) & (vi < H - off) & (vj >= off) & (vj < W - off)
        vi, vj = vi[keep], vj[keep]

    if vi.size == 0:
        raise ValueError("No valid pixels found for inference.")

    # ---- 5) Features extraction according to model type ----
    if model_type == "pixel":
        X = series_pad[:, vi, vj].transpose(1, 0)[:, :, None]         # (N, T_eff, 1)
    else:
        off = patch_size // 2
        patches = [
            series_pad[:, vi + di, vj + dj]
            for di in range(-off, off + 1)
            for dj in range(-off, off + 1)
        ]
        X = np.stack(patches, axis=-1).transpose(1, 0, 2)             # (N, T_eff, P)

    # ---- 6) Predictions in proba ----
    probs = predict_on_subseries(
        model, X, device=device, smooth=smooth_temporal,
        batch_size=batch_size, return_proba=True
    )[:, :T_eff]

    # ---- 7) Binarization of predictions + duration constraint ----
    y_bin = (probs >= float(threshold)).astype(np.uint8)
    if min_duration and min_duration > 1:
        y_bin = enforce_min_duration(y_bin, min_duration)  # (N, T_eff)

    # ---- 8) Reprojection spatio-temporelle ----
    prob_cube = np.zeros((T_eff, H, W), dtype=np.float32)
    bin_cube  = np.zeros((T_eff, H, W), dtype=bool)
    for t in range(T_eff):
        prob_cube[t, vi, vj] = probs[:, t]
        bin_cube[t,  vi, vj] = y_bin[:, t].astype(bool)

    # ---- 9) Gaussian filter + removing of smaller objects ----
    masks_bin = np.zeros_like(bin_cube)
    for t in range(T_eff):
        frame_bin = bin_cube[t]
        if sigma and sigma > 0:
            smoothed = gaussian_filter(prob_cube[t], sigma=sigma)
            frame_bin = smoothed >= threshold
        if min_size and min_size > 0:
            frame_bin = remove_small_objects(frame_bin, min_size=min_size)
        masks_bin[t] = frame_bin

    # ---- 10) Elimination of padding frames ----
    masks_bin = masks_bin[:T_orig]  # (T_orig, H, W) bool

    # ---- 11) Saving of images in folder <image_dir>/SEG_<model_type> ----
    outdir = Path(image_dir).parent / folder_name
    outdir.mkdir(parents=True, exist_ok=True)
    for t in range(T_orig):
        arr = (masks_bin[t].astype(np.uint8) * 255)
        img = Image.fromarray(arr, mode="L")
        if seg_mode == "1":
            img = img.convert("1")
        img.save(outdir / f"{seg_prefix}{t+1:04d}.png")

    print(f"{T_orig} segmentations saved in folder: {outdir}")
    return str(outdir)

def segment_series_distance(
    model,
    T: int = 16,
    device: str = "cpu",
    threshold: float = 0.5,
    min_duration: int = 3,
    min_size: int = 20,
    sigma: float = 0.0,
    smooth_temporal: bool = False,
    batch_size: int = 8192,
    seg_prefix: str = "seg_",
    folder_name: str = "SEG_distance",
    seg_mode: str = "L",
):
    """
    Inférence pixel-only avec vecteurs [intensité, distance (non normalisée)].
    - mask_path: PNG binaire même taille (H×W) que les images.
    - Entrée modèle: (N, T, 2).
    Renvoie (liste_paths, stack_binaire_TxHxW).
    """

    # --- Selection window ---

    root = tk.Tk(); root.withdraw()
    image_dir = filedialog.askdirectory(title="Select images folder")
    if not image_dir:
        raise RuntimeError("No selected folder")
    
    mask_path = filedialog.askopenfilename(
        title="Select a GA binary mask, must be same size as the IR images",
        filetypes=[("PNG","*.png"), ("Tous","*.*")])
    if not mask_path: 
        raise RuntimeError("Sélection du masque annulée.")

    # --- Loading of images and mask ---
        
    series = load_ir_images(image_dir)  
    if series.ndim != 3:
        raise ValueError(f"load_ir_images must be of shape (T,H,W), here {series.shape}")
    T0, H, W = series.shape

    mask_arr = np.array(Image.open(mask_path).convert("L"))
    if mask_arr.shape != (H, W):
        raise ValueError(f"Masque {mask_arr.shape} différent des images {(H,W)}.")
    GA_mask = (mask_arr > 0)  # bool

    # --- crop/pad temporel ---
    if T0 < T:
        pad = T - T0
        series_pad = np.concatenate([series, np.zeros((pad, H, W), dtype=series.dtype)], axis=0)
    else:
        series_pad = series[:T]
    T_eff = series_pad.shape[0]

    # --- pixels valides (toute intensité >0 au moins une fois) ---
    valid = compute_validity_mask(series_pad, threshold=0)
    vi, vj = np.where(valid)
    if vi.size == 0:
        raise ValueError("Aucun pixel valide trouvé.")
    coords = np.stack([vi, vj], axis=1)
    N = coords.shape[0]

    # --- distance NON normalisée depuis le GA final (0 sur GA, augmente en dehors) ---
    dist_map = distance_transform_edt(~GA_mask).astype(np.float32)

       # --- Construire X (N,T,2) = [intensité, distance_constante] ---
    N = vi.size
    X = np.empty((N, T_eff, 2), dtype=np.float32)
    for k, (i, j) in enumerate(zip(vi, vj)):
        X[k, :, 0] = series_pad[:, i, j].astype(np.float32)  # intensité
        X[k, :, 1] = dist_map[i, j]                          # distance (constante sur T)

    # --- Inférence  ---
    probs = predict_on_subseries(
        model, X, device=device, smooth=smooth_temporal,
        batch_size=batch_size, return_proba=True
    )[:, :T_eff]  # (N,T_eff)

    # --- Binarisation + min_duration ---
    y_bin = (probs >= float(threshold)).astype(np.uint8)
    if min_duration and min_duration > 1:
        y_bin = enforce_min_duration(y_bin, min_duration)  # (N,T_eff)

    # --- Reprojection spatio-temporelle ---
    prob_cube = np.zeros((T_eff, H, W), dtype=np.float32)
    bin_cube  = np.zeros((T_eff, H, W), dtype=bool)
    for t in range(T_eff):
        prob_cube[t, vi, vj] = probs[:, t]
        bin_cube[t,  vi, vj] = y_bin[:, t].astype(bool)

    # --- Lissage spatial + min_size ---
    masks_bin = np.zeros_like(bin_cube)
    for t in range(T_eff):
        frame_bin = bin_cube[t]
        if sigma and sigma > 0:
            smoothed = gaussian_filter(prob_cube[t], sigma=sigma)
            frame_bin = smoothed >= threshold
        if min_size and min_size > 0:
            frame_bin = remove_small_objects(frame_bin, min_size=min_size)
        masks_bin[t] = frame_bin

    # --- Supprimer frames de padding ---
    masks_bin = masks_bin[:T0+1]

    # --- Sauvegarde ---
    outdir = Path(image_dir).parent / folder_name   # folder_name: str
    outdir.mkdir(parents=True, exist_ok=True)

    for t in range(T0):
        arr = (masks_bin[t].astype(np.uint8) * 255)
        # Évite le warning Pillow: on convertit après
        img = Image.fromarray(arr).convert("L")
        if seg_mode == "1":
            img = img.convert("1")
        img.save(outdir / f"{seg_prefix}{t+1:04d}.png")

    print(f"{T0} segmentations saved in folder: {outdir}")

    # ➜ Retourne uniquement le chemin du dossier (string)
    return str(outdir)

def segment_series_to_colormap(
    model,
    model_type: str = "pixel",         # "pixel" ou "patch"
    T: int = 16,
    device: str = "cpu",
    threshold: float = 0.5,
    min_duration: int = 3,
    min_size: int = 20,
    sigma: float = 0.0,
    smooth_temporal: bool = False,
    batch_size: int = 1024,
    seg_prefix: str = "seg_",
    folder_name: str = "SEG_",
    stride: int = 8,
    cmap_name: str = "turbo",          # << nom de la colormap
    absolute_index: bool = False,      # False: index relatif à la fenêtre ; True: index absolu global
    save_indices_tiff: bool = True,    # Sauver aussi la carte d'indices (int16)
):
    """
    Segmente en sous-séries chevauchantes et sauvegarde UNE colormap d'onset par sous-série.
    - La couleur encode l'instant (frame) où le pixel passe en GA (premier 1 après filtres).
    - Les pixels sans apparition gardent l'alpha 0 (transparent) et sont -1 dans le TIFF.
    """
    assert model_type in ("pixel", "patch")
    assert stride >= 1

    # ---- 1) Sélection & chargement ----
    root = tk.Tk(); root.withdraw()
    image_dir = filedialog.askdirectory(title="Select images folder")
    if not image_dir:
        raise RuntimeError("No selected folder")
    series_full = load_ir_images(image_dir)  # (T0,H,W) normalisées
    if series_full.ndim != 3:
        raise ValueError(f"load_ir_images doit renvoyer (T,H,W), obtenu {series_full.shape}")

    T0, H, W = series_full.shape

    # ---- 2) Dossier racine de sortie ----
    out_root = Path(image_dir).parent / folder_name
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 3) Fenêtres (garantir >= stride frames utiles sur la dernière) ----
    starts = [0]
    while starts[-1] + T < T0:
        nxt = starts[-1] + stride
        if nxt >= T0: break
        starts.append(nxt)
    if starts[-1] + T < T0 and (T0 - stride) > starts[-1]:
        starts.append(T0 - stride)

    # Pré-colormap
    try:
        cmap = cm.get_cmap(cmap_name)
    except Exception as e:
        raise ValueError(f"Colormap '{cmap_name}' introuvable dans Matplotlib.") from e

    total_saved = 0

    for w_idx, s in enumerate(starts, start=1):
        e = min(s + T, T0)
        sub = series_full[s:e]      # (L,H,W)
        L = sub.shape[0]

        # Padding pour constituer des fenêtres de taille T pour le modèle
        if L < T:
            pad = np.zeros((T - L, H, W), dtype=sub.dtype)
            series_pad = np.concatenate([sub, pad], axis=0)   # (T,H,W)
        else:
            series_pad = sub

        # ---- Validité & bords (par fenêtre) ----
        mask_valid = compute_validity_mask(series_pad, threshold=0)  # (H,W)
        vi, vj = np.where(mask_valid)

        if model_type == "patch":
            patch_size = 3
            off = patch_size // 2
            keep = (vi >= off) & (vi < H - off) & (vj >= off) & (vj < W - off)
            vi, vj = vi[keep], vj[keep]

        if vi.size == 0:
            # crée quand même le dossier seg et passe
            (out_root / f"seg{w_idx}").mkdir(parents=True, exist_ok=True)
            continue

        # ---- Features ----
        if model_type == "pixel":
            X = series_pad[:, vi, vj].transpose(1, 0)[:, :, None].astype(np.float32)  # (N,T,1)
        else:
            off = patch_size // 2
            patches = [
                series_pad[:, vi + di, vj + dj]
                for di in range(-off, off + 1)
                for dj in range(-off, off + 1)
            ]
            X = np.stack(patches, axis=-1).transpose(1, 0, 2).astype(np.float32)      # (N,T,P)

        # ---- Probas (N,T) -> ne garder que L frames utiles ----
        probs = predict_on_subseries(
            model, X, device=device, smooth=smooth_temporal,
            batch_size=batch_size, return_proba=True
        )[:, :L]  # (N,L)

        # ---- Lissage spatial optionnel AVANT seuillage (frame-wise) ----
        if sigma and sigma > 0:
            # Reprojeter temporairement pour filtrer en 2D
            prob_cube = np.zeros((L, H, W), dtype=np.float32)
            for t_rel in range(L):
                prob_cube[t_rel, vi, vj] = probs[:, t_rel]
            prob_cube = gaussian_filter(prob_cube, sigma=(0, sigma, sigma))
            # Retour (N,L)
            probs = prob_cube[:, vi, vj].T

        # ---- Seuillage & min_duration (par pixel, sur la fenêtre) ----
        y_bin = (probs >= float(threshold)).astype(np.uint8)  # (N,L)
        if min_duration and min_duration > 1:
            y_bin = enforce_min_duration(y_bin, min_duration)  # (N,L)

        # ---- Suppression petites composantes (frame-wise) ----
        if min_size and min_size > 0:
            # Reprojeter pour opérer par frame, puis re-extraire
            bin_cube = np.zeros((L, H, W), dtype=bool)
            for t_rel in range(L):
                bin_cube[t_rel, vi, vj] = y_bin[:, t_rel].astype(bool)
                bin_cube[t_rel] = remove_small_objects(bin_cube[t_rel], min_size=min_size)
            y_bin = bin_cube[:, vi, vj].T  # (N,L) bool

        # ---- Calcul de l'ONSET (premier True) ----
        # has_onset: (N,), first_idx: (N,), retourne 0 si pas de True ⇒ attention
        has_onset = y_bin.any(axis=1)
        first_idx = y_bin.argmax(axis=1)  # premier index de True OU 0 si aucun

        # Construire la carte d'indices (H,W) avec -1 = pas d'apparition
        onset_idx = np.full((H, W), -1, dtype=np.int16)

        if absolute_index:
            # index absolu dans la série globale
            onset_vals = (s + first_idx[has_onset]).astype(np.int32)
            onset_idx[vi[has_onset], vj[has_onset]] = onset_vals
            norm_den = max(T0 - 1, 1)
            norm = onset_vals.astype(np.float32) / norm_den
            # Pour normaliser toute l'image, on créera plus bas une carte normalisée complète
        else:
            # index relatif à la fenêtre [0..L-1]
            onset_vals = first_idx[has_onset].astype(np.int32)
            onset_idx[vi[has_onset], vj[has_onset]] = onset_vals
            norm_den = max(L - 1, 1)

        # ---- Conversion en image couleur (RGBA), pixels “sans apparition” transparents ----
        # Carte normalisée 0..1 pour les pixels ayant une apparition :
        onset_norm = np.zeros((H, W), dtype=np.float32)
        if absolute_index:
            # Normaliser par la longueur globale
            valid_abs = onset_idx >= 0
            onset_norm[valid_abs] = (onset_idx[valid_abs].astype(np.float32)) / max(T0 - 1, 1)
        else:
            valid_rel = onset_idx >= 0
            onset_norm[valid_rel] = (onset_idx[valid_rel].astype(np.float32)) / norm_den

        rgba = np.zeros((H, W, 4), dtype=np.float32)
        if (onset_norm > 0).any() or (onset_idx == 0).any():
            rgba_colors = cmap(onset_norm)  # (H,W,4), alpha=1 par défaut
            rgba[onset_idx >= 0] = rgba_colors[onset_idx >= 0]
            rgba[onset_idx >= 0, 3] = 1.0   # alpha plein sur pixels avec apparition
            rgba[onset_idx <  0, 3] = 0.0   # transparent ailleurs

        # ---- Sauvegarde dans seg{w_idx} ----
        seg_dir = out_root / f"seg{w_idx}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # 1) PNG couleur (RGBA) de la colormap
        # Convertir en 8-bit
        rgba8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgba8, mode="RGBA").save(seg_dir / "onset_colormap.png")

        # 2) (optionnel) TIFF d'indices (int16) pour analyses
        if save_indices_tiff:
            # Sauve en TIFF grayscale 16 bits signé : via Pillow -> utiliser mode 'I;16' nécessite non signé.
            # On convertit en int16, puis décalage +32768 pour stocker en uint16 (restaurable).
            onset_u16 = (onset_idx.astype(np.int32) + 32768).astype(np.uint16)
            Image.fromarray(onset_u16, mode="I;16").save(seg_dir / "onset_indices.tiff")

        total_saved += 1

    print(f"{total_saved} onset colormaps saved in folder: {out_root}")
    return str(out_root)

## After Processing and helpers

def pngs_to_mat(input_folder, output_matfile, var_name="MASK"):

    """This function takes as input a folder with n png binary images of size (H,W)
      and transforms them into a unique matlab folder of size (n, H, W)"""
    
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    first_image = Image.open(os.path.join(input_folder, image_files[0])).convert("L")
    height, width = first_image.size[::-1]
    num_images = len(image_files)
    image_stack = np.zeros((num_images, height, width), dtype=np.uint8)

    for i, file in enumerate(image_files):
        img = Image.open(os.path.join(input_folder, file)).convert("L")
        binary = np.array(img) > 127  # binaire 0/1
        image_stack[i] = binary.astype(np.uint8)

    print("Final shape (T, H, W):", image_stack.shape)
    savemat(output_matfile, {var_name: image_stack})
    print(f"Saved {num_images} images to '{output_matfile}' as variable '{var_name}'.")



