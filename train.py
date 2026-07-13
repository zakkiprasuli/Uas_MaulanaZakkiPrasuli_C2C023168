"""
train.py
=========
Script untuk MELATIH ULANG eksperimen pada paper: membandingkan performa
CNN (U-Net) pada 5 skenario preprocessing (original, HE, CLAHE, HE-CLAHE,
CLAHE-HE) menggunakan dataset Lung CT-Scan atau Chest X-ray (format Kaggle
yang disebut pada paper).

STRUKTUR DATASET YANG DIHARAPKAN:
    dataset/
        images/       *.png / *.jpg   (citra asli)
        masks/        *.png / *.jpg   (ground truth, nama file sama)

Dataset referensi paper (unduh manual dari Kaggle, lalu susun sesuai
struktur di atas):
    - Lung CT-Scan : kaggle.com/datasets/kmader/finding-lungs-in-ct-data
    - Chest X-ray  : kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

CARA PAKAI:
    python train.py --data_dir dataset/ --method clahe_he --epochs 30 --max_samples 3616

Setelah training, model disimpan ke models/<method>.h5 dan metrik
(loss, accuracy, mse, dsc, ssim) dicetak & disimpan ke models/results.json
sesuai format Tabel 1 pada paper.
"""

import os
import json
import argparse
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from enhancement import apply_method, to_8bit          # noqa: E402
from model import build_unet, compile_model            # noqa: E402
from metrics import dice_similarity_coefficient, ssim_score  # noqa: E402


IMG_SIZE = 256


def load_dataset(data_dir, method, cdf_min=0, cdf_max=39, clip_limit=0.01, max_samples=None, seed=42):
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    # Cocokkan berdasarkan nama file TANPA ekstensi, konsisten dengan check_pairs.py,
    # supaya mis. "foo.png" (image) tetap terpasang dengan "foo.jpg" (mask).
    img_map = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}
    mask_map = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}
    stems = sorted(set(img_map.keys()) & set(mask_map.keys()))

    print(f"      Ditemukan {len(stems)} pasangan citra+mask yang valid dari "
          f"{len(img_map)} images / {len(mask_map)} masks.")

    if max_samples is not None and len(stems) > max_samples:
        rng = np.random.default_rng(seed)
        stems = list(rng.choice(stems, size=max_samples, replace=False))
        print(f"      Disubsample acak (seed={seed}) menjadi {max_samples} data "
              f"(menyamai skala dataset pada paper).")

    X, Y = [], []
    skipped_unreadable = 0
    for stem in stems:
        img_path = os.path.join(img_dir, img_map[stem])
        mask_path = os.path.join(mask_dir, mask_map[stem])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skipped_unreadable += 1
            continue

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = to_8bit(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img = apply_method(method, img, cdf_min=cdf_min, cdf_max=cdf_max, clip_limit=clip_limit)

        X.append(img.astype(np.float32) / 255.0)
        Y.append((mask > 127).astype(np.float32))

    if skipped_unreadable:
        print(f"      Dilewati (gagal dibaca / format tidak didukung, mis. .nii.gz): {skipped_unreadable}")

    X = np.expand_dims(np.array(X), -1)
    Y = np.expand_dims(np.array(Y), -1)
    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path ke folder dataset (images/, masks/)")
    parser.add_argument("--method", default="clahe_he",
                         choices=["original", "he", "clahe", "he_clahe", "clahe_he"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cdf_min", type=int, default=0)
    parser.add_argument("--cdf_max", type=int, default=39)
    parser.add_argument("--clip_limit", type=float, default=0.01)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Batasi jumlah data (subsample acak). Contoh: 3616 untuk menyamai "
                              "skala dataset Chest X-ray pada paper, 267 untuk Lung CT-Scan.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk subsampling")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] Memuat dataset & menerapkan preprocessing '{args.method}' ...")
    X, Y = load_dataset(args.data_dir, args.method, args.cdf_min, args.cdf_max, args.clip_limit,
                         max_samples=args.max_samples, seed=args.seed)
    print(f"      Total data: {len(X)} citra")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("[2/4] Membangun model U-Net ...")
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    model = compile_model(model, learning_rate=args.lr)

    print("[3/4] Training ...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    model_path = os.path.join(args.out_dir, f"{args.method}.h5")
    model.save(model_path)
    print(f"      Model disimpan: {model_path}")

    print("[4/4] Evaluasi (Loss, Accuracy, MSE, DSC, SSIM) ...")
    train_eval = model.evaluate(X_train, Y_train, verbose=0)
    test_eval = model.evaluate(X_test, Y_test, verbose=0)

    preds = model.predict(X_test, verbose=0)
    dsc_list, ssim_list = [], []
    for p, gt in zip(preds, Y_test):
        p_bin = (p[..., 0] * 255).astype(np.uint8)
        gt_bin = (gt[..., 0] * 255).astype(np.uint8)
        dsc_list.append(dice_similarity_coefficient(gt_bin, p_bin))
        ssim_list.append(ssim_score(gt_bin, p_bin))

    result = {
        "method": args.method,
        "train": {"loss": train_eval[0], "accuracy": train_eval[1], "mse": train_eval[2]},
        "test": {"loss": test_eval[0], "accuracy": test_eval[1], "mse": test_eval[2]},
        "dsc_mean": float(np.mean(dsc_list)),
        "ssim_mean": float(np.mean(ssim_list)),
    }

    results_path = os.path.join(args.out_dir, "results.json")
    all_results = {}
    if os.path.exists(results_path):
        all_results = json.load(open(results_path))
    all_results[args.method] = result
    json.dump(all_results, open(results_path, "w"), indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()