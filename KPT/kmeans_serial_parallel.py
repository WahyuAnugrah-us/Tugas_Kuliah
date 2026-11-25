"""
Praktikum Komputasi Paralel: K-Means Serial vs Paralel (Full Benchmark & Tabel Laporan)
Dataset: dataset_kmeans_1000000.csv
Fitur clustering: Sales, Profit, Quantity, Price
Kolom true_cluster hanya label pembentuk data sintetis untuk referensi,
TIDAK digunakan sebagai fitur K-Means.

Konfigurasi Pengujian:
- Serial  : 1 proses
- Paralel : 2, 4, 8, dan 12 proses (sesuai kapasitas 12 logical threads Intel Core i5-12450H)

Output otomatis (CSV & Ringkasan Terminal):
1. tabel_spesifikasi_perangkat.csv   -> Spesifikasi hardware & environment otomatis
2. hasil_benchmark_final.csv         -> Hasil benchmark utama 1.000.000 data (1, 2, 4, 8, 12 proses)
3. tabel_perbandingan_optimasi.csv   -> Perbandingan Sebelum vs Sesudah Optimasi
4. tabel_statistik_klaster.csv       -> Statistik rata-rata fitur per klaster
5. tabel_skalabilitas_dataset.csv    -> Uji skalabilitas 100k, 250k, 500k, 1M baris (1, 2, 4, 8, 12 proses)
"""

import os

# Wajib diletakkan sebelum import numpy agar tidak terjadi thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from multiprocessing import Pool, freeze_support
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import platform
import socket
import sys
import time
import numpy as np
import pandas as pd

FEATURES = ["Sales", "Profit", "Quantity", "Price"]
N_CLUSTERS = 4
MAX_ITER = 100
TOL = 1e-4
SEED = 123
PARALLEL_PROCESS_COUNTS = (2, 4, 8, 12)

# Variabel global pada worker
_WORKER_X = None
_WORKER_SHM = None


def load_dataset(nrows=None):
    path = Path(__file__).with_name("dataset_kmeans_1000000.csv")
    df = pd.read_csv(path, nrows=nrows)
    X = df[FEATURES].to_numpy(dtype=np.float64)
    return X, df


def initialize_centroids(X, k=N_CLUSTERS, seed=SEED):
    """Memilih centroid awal secara acak; seed membuat eksperimen reproducible."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


# ==============================================================================
# 1. VERSI SEBELUM OPTIMASI (Baseline Awal: Copy Dataset + np.add.at + Return Labels)
# ==============================================================================
def _assign_and_summarize_unoptimized(X_chunk, centroids):
    distances = np.sum((X_chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    k = len(centroids)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    sums = np.zeros((k, X_chunk.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, X_chunk)
    return labels, counts, sums


def _init_worker_copy(X):
    global _WORKER_X
    _WORKER_X = X


def _worker_unoptimized(task):
    start, end, centroids = task
    return _assign_and_summarize_unoptimized(_WORKER_X[start:end], centroids)


def kmeans_serial_unoptimized(X, initial_centroids):
    centroids = initial_centroids.copy()
    for iteration in range(1, MAX_ITER + 1):
        labels, counts, sums = _assign_and_summarize_unoptimized(X, centroids)
        new_centroids = centroids.copy()
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]
        if np.linalg.norm(new_centroids - centroids) <= TOL:
            centroids = new_centroids
            break
        centroids = new_centroids
    return labels, centroids, iteration


def kmeans_parallel_unoptimized(X, initial_centroids, n_processes=4):
    centroids = initial_centroids.copy()
    n = len(X)
    boundaries = np.linspace(0, n, n_processes + 1, dtype=int)
    ranges = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_processes)]

    with Pool(processes=n_processes, initializer=_init_worker_copy, initargs=(X,)) as pool:
        for iteration in range(1, MAX_ITER + 1):
            tasks = [(s, e, centroids) for s, e in ranges]
            partial_results = pool.map(_worker_unoptimized, tasks)

            total_counts = np.zeros(N_CLUSTERS, dtype=np.int64)
            total_sums = np.zeros((N_CLUSTERS, X.shape[1]), dtype=np.float64)
            labels_parts = []
            for lbl_p, cnt_p, sum_p in partial_results:
                labels_parts.append(lbl_p)
                total_counts += cnt_p
                total_sums += sum_p

            labels = np.concatenate(labels_parts)
            new_centroids = centroids.copy()
            non_empty = total_counts > 0
            new_centroids[non_empty] = total_sums[non_empty] / total_counts[non_empty, None]
            if np.linalg.norm(new_centroids - centroids) <= TOL:
                centroids = new_centroids
                break
            centroids = new_centroids

    return labels, centroids, iteration


# ==============================================================================
# 2. VERSI SESUDAH OPTIMASI (Zero-Copy SharedMemory + Vectorized np.bincount)
# ==============================================================================
def assign_labels(X_chunk, centroids):
    distances = np.sum((X_chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def summarize_chunk(X_chunk, centroids):
    labels = assign_labels(X_chunk, centroids)
    k = len(centroids)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    sums = np.column_stack([
        np.bincount(labels, weights=X_chunk[:, j], minlength=k)
        for j in range(X_chunk.shape[1])
    ])
    return counts, sums


def kmeans_serial_optimized(X, initial_centroids):
    centroids = initial_centroids.copy()
    for iteration in range(1, MAX_ITER + 1):
        counts, sums = summarize_chunk(X, centroids)
        new_centroids = centroids.copy()
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]
        if np.linalg.norm(new_centroids - centroids) <= TOL:
            centroids = new_centroids
            break
        centroids = new_centroids
    labels = assign_labels(X, centroids)
    return labels, centroids, iteration


def _init_worker_shm(shm_name, shape, dtype_str):
    global _WORKER_X, _WORKER_SHM
    _WORKER_SHM = SharedMemory(name=shm_name)
    _WORKER_X = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_WORKER_SHM.buf)


def _worker_summarize_shm(task):
    start, end, centroids = task
    return summarize_chunk(_WORKER_X[start:end], centroids)


def _worker_labels_shm(task):
    start, end, centroids = task
    return assign_labels(_WORKER_X[start:end], centroids)


def kmeans_parallel_optimized(X, initial_centroids, n_processes=4):
    centroids = initial_centroids.copy()
    n = len(X)
    boundaries = np.linspace(0, n, n_processes + 1, dtype=int)
    ranges = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_processes)]

    shm = SharedMemory(create=True, size=X.nbytes)
    try:
        shm_array = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
        shm_array[:] = X

        with Pool(
            processes=n_processes,
            initializer=_init_worker_shm,
            initargs=(shm.name, X.shape, X.dtype.str),
        ) as pool:
            for iteration in range(1, MAX_ITER + 1):
                tasks = [(s, e, centroids) for s, e in ranges]
                partials = pool.map(_worker_summarize_shm, tasks)

                total_counts = np.zeros(N_CLUSTERS, dtype=np.int64)
                total_sums = np.zeros((N_CLUSTERS, X.shape[1]), dtype=np.float64)
                for cnt_p, sum_p in partials:
                    total_counts += cnt_p
                    total_sums += sum_p

                new_centroids = centroids.copy()
                non_empty = total_counts > 0
                new_centroids[non_empty] = total_sums[non_empty] / total_counts[non_empty, None]
                if np.linalg.norm(new_centroids - centroids) <= TOL:
                    centroids = new_centroids
                    break
                centroids = new_centroids

            label_tasks = [(s, e, centroids) for s, e in ranges]
            labels = np.concatenate(pool.map(_worker_labels_shm, label_tasks))
    finally:
        shm.close()
        shm.unlink()

    return labels, centroids, iteration


# ==============================================================================
# 3. FUNGSI PEMBANGKIT SELURUH TABEL LAPORAN & PPT
# ==============================================================================
def generate_hardware_table(out_dir):
    """Tabel 1: Spesifikasi Perangkat Keras & Lingkungan."""
    rows = [
        {"Komponen": "Device Name", "Spesifikasi_Teknis": socket.gethostname()},
        {"Komponen": "Sistem Operasi", "Spesifikasi_Teknis": f"{platform.system()} {platform.release()} ({platform.machine()})"},
        {"Komponen": "Processor", "Spesifikasi_Teknis": platform.processor() or "Unknown CPU"},
        {"Komponen": "Logical Threads", "Spesifikasi_Teknis": f"{os.cpu_count()} Logical Processors"},
        {"Komponen": "Versi Python", "Spesifikasi_Teknis": sys.version.split()[0]},
        {"Komponen": "Versi NumPy / Pandas", "Spesifikasi_Teknis": f"NumPy {np.__version__} / Pandas {pd.__version__}"},
    ]
    df_hw = pd.DataFrame(rows)
    df_hw.to_csv(out_dir / "tabel_spesifikasi_perangkat.csv", index=False)
    print("\n[1/5] TABEL SPESIFIKASI PERANGKAT:")
    print(df_hw.to_string(index=False))
    return df_hw


def generate_cluster_stats_table(df, labels, out_dir):
    """Tabel 2: Statistik Rata-rata Fitur per Klaster Hasil Clustering."""
    df_temp = df[FEATURES].copy()
    df_temp["Cluster"] = labels
    order = df_temp.groupby("Cluster")["Sales"].mean().sort_values().index.tolist()
    mapping = {old_c: new_c for new_c, old_c in enumerate(order)}
    df_temp["Cluster"] = df_temp["Cluster"].map(mapping)

    summary = (
        df_temp.groupby("Cluster")
        .agg(
            Jumlah_Data=("Sales", "count"),
            Mean_Sales=("Sales", "mean"),
            Mean_Profit=("Profit", "mean"),
            Mean_Quantity=("Quantity", "mean"),
            Mean_Price=("Price", "mean"),
        )
        .reset_index()
    )
    summary["Cluster"] = summary["Cluster"].apply(lambda c: f"Cluster {c}")
    summary = summary.round(2)
    summary.to_csv(out_dir / "tabel_statistik_klaster.csv", index=False)
    print("\n[2/5] TABEL STATISTIK RATA-RATA KLASTER:")
    print(summary.to_string(index=False))
    return summary


def run_benchmark_table(X, serial_fn, parallel_fn, repeats=3, process_counts=PARALLEL_PROCESS_COUNTS):
    """Menjalankan benchmark untuk konfigurasi 1 (Serial) dan 2, 4, 8, 12 (Paralel)."""
    initial = initialize_centroids(X)

    serial_times = []
    serial_res = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        serial_res = serial_fn(X, initial)
        serial_times.append(time.perf_counter() - t0)

    t_serial = float(np.mean(serial_times))
    rows = [{
        "Metode": "Serial",
        "Jumlah_Proses": 1,
        "Rata_rata_waktu_detik": t_serial,
        "Speedup": 1.0,
        "Efficiency_persen": 100.0,
        "Iterasi": serial_res[2],
        "Centroid_Mendekati_Sama": True,
    }]

    max_cpu = os.cpu_count() or 1
    for p in process_counts:
        if p > max_cpu:
            print(f"Lewati {p} proses: melebihi logical processors ({max_cpu}).")
            continue
        p_times = []
        p_res = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            p_res = parallel_fn(X, initial, n_processes=p)
            p_times.append(time.perf_counter() - t0)

        t_par = float(np.mean(p_times))
        speedup = t_serial / t_par
        efficiency = (speedup / p) * 100.0
        same_centroids = bool(np.allclose(serial_res[1], p_res[1], atol=1e-3))

        rows.append({
            "Metode": "Paralel",
            "Jumlah_Proses": p,
            "Rata_rata_waktu_detik": t_par,
            "Speedup": speedup,
            "Efficiency_persen": efficiency,
            "Iterasi": p_res[2],
            "Centroid_Mendekati_Sama": same_centroids,
        })

    return pd.DataFrame(rows), serial_res[0]


def generate_optimization_and_final_tables(X, out_dir, repeats=3, process_counts=PARALLEL_PROCESS_COUNTS):
    """Tabel 3 & Tabel 4: Perbandingan Sebelum vs Sesudah Optimasi + Hasil Final (1, 2, 4, 8, 12 proses)."""
    print("\nMenjalankan benchmark SEBELUM optimasi (1, 2, 4, 8, 12 proses)...")
    df_before, _ = run_benchmark_table(
        X, kmeans_serial_unoptimized, kmeans_parallel_unoptimized, repeats=repeats, process_counts=process_counts
    )

    print("Menjalankan benchmark SESUDAH optimasi (SharedMemory + bincount: 1, 2, 4, 8, 12 proses)...")
    df_after, final_labels = run_benchmark_table(
        X, kmeans_serial_optimized, kmeans_parallel_optimized, repeats=repeats, process_counts=process_counts
    )

    # Simpan Tabel Hasil Benchmark Final
    df_after.to_csv(out_dir / "hasil_benchmark_final.csv", index=False)
    print("\n[3/5] TABEL HASIL BENCHMARK FINAL (1.000.000 BARIS):")
    print(df_after.to_string(index=False))

    # Gabungkan untuk Tabel Perbandingan Sebelum vs Sesudah Optimasi
    comp_rows = []
    for idx in range(len(df_after)):
        metode = df_after.loc[idx, "Metode"]
        procs = int(df_after.loc[idx, "Jumlah_Proses"])
        t_awal = df_before.loc[idx, "Rata_rata_waktu_detik"]
        t_opt = df_after.loc[idx, "Rata_rata_waktu_detik"]
        hemat_pct = ((t_opt - t_awal) / t_awal) * 100.0
        comp_rows.append({
            "Konfigurasi": f"{metode} ({procs} Proses)",
            "Waktu_Awal_Detik": round(t_awal, 4),
            "Waktu_Optimasi_Detik": round(t_opt, 4),
            "Perubahan_Waktu_Persen": f"{hemat_pct:.1f}%",
            "Speedup_Optimasi": round(df_after.loc[idx, "Speedup"], 4),
        })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(out_dir / "tabel_perbandingan_optimasi.csv", index=False)
    print("\n[4/5] TABEL PERBANDINGAN SEBELUM VS SESUDAH OPTIMASI:")
    print(df_comp.to_string(index=False))

    return df_after, df_comp, final_labels


def generate_scalability_table(
    X_full,
    out_dir,
    sizes=(100_000, 250_000, 500_000, 1_000_000),
    repeats=2,
    process_counts=PARALLEL_PROCESS_COUNTS,
):
    """Tabel 5: Uji Skalabilitas di 4 Ukuran Dataset (100k, 250k, 500k, 1M) untuk 1, 2, 4, 8, 12 proses."""
    all_rows = []
    for n_rows in sizes:
        if n_rows > len(X_full):
            continue
        X_sub = X_full[:n_rows]
        df_sub, _ = run_benchmark_table(
            X_sub, kmeans_serial_optimized, kmeans_parallel_optimized, repeats=repeats, process_counts=process_counts
        )
        df_sub.insert(0, "Jumlah_Baris", n_rows)
        all_rows.append(df_sub)

    df_scale = pd.concat(all_rows, ignore_index=True)
    df_scale.to_csv(out_dir / "tabel_skalabilitas_dataset.csv", index=False)
    print("\n[5/5] TABEL SKALABILITAS MULTI-UKURAN DATASET:")
    print(df_scale.to_string(index=False))
    return df_scale


if __name__ == "__main__":
    freeze_support()
    out_dir = Path(__file__).resolve().parent

    print("Memuat dataset_kmeans_1000000.csv...")
    X, df = load_dataset()
    print(f"Dataset dimuat: {len(X):,} baris x {X.shape[1]} fitur ({', '.join(FEATURES)})")
    print(f"Konfigurasi proses uji: 1 (Serial), {', '.join(map(str, PARALLEL_PROCESS_COUNTS))} (Paralel)")

    # 1. Tabel Spesifikasi Perangkat Keras & Environment
    generate_hardware_table(out_dir)

    # 2 & 3. Tabel Hasil Benchmark Final & Tabel Perbandingan Sebelum vs Sesudah Optimasi
    _, _, final_labels = generate_optimization_and_final_tables(
        X, out_dir, repeats=3, process_counts=PARALLEL_PROCESS_COUNTS
    )

    # 4. Tabel Statistik Rata-rata Klaster Hasil Clustering
    generate_cluster_stats_table(df, final_labels, out_dir)

    # 5. Tabel Skalabilitas (100k, 250k, 500k, 1M baris)
    generate_scalability_table(
        X, out_dir, sizes=(100_000, 250_000, 500_000, 1_000_000), repeats=2, process_counts=PARALLEL_PROCESS_COUNTS
    )

    print("\nSelesai! Seluruh 5 file tabel CSV berhasil disimpan di folder yang sama dengan skrip.")
