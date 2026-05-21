import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120

# =============================================================================
# 0. KONFIGURASI
# =============================================================================
API_KEY    = "e89377b8d9c451a43ebc5bb42a9fe555"
BASE_URL   = "http://ws.audioscrobbler.com/2.0/"
OUTPUT_RAW = "PramesaAlyusufi_data_lastfmaudio_teks_API.csv"
OUTPUT_FE  = "PramesaAlyusufi_data_lastfmaudio_feature_engineered.csv"
IMG_DIR    = "visualisasi"
os.makedirs(IMG_DIR, exist_ok=True)

GENRES           = ["pop", "rock", "jazz", "electronic", "hip-hop",
                    "classical", "indie", "metal", "rnb", "country"]
TRACKS_PER_GENRE = 100

# =============================================================================
# 1. PENGUMPULAN DATA — LAST.FM API (FIELD LENGKAP)
# =============================================================================
def ambil_top_tracks(genre: str, limit: int = 100) -> list:
    params = {
        "method"  : "tag.gettoptracks",
        "tag"     : genre,
        "api_key" : API_KEY,
        "format"  : "json",
        "limit"   : limit
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        return r.json().get("tracks", {}).get("track", [])
    except Exception as e:
        print(f"  [ERROR] Genre {genre}: {e}")
        return []


def ambil_detail_track(artist: str, track: str) -> dict:
    params = {
        "method"  : "track.getInfo",
        "artist"  : artist,
        "track"   : track,
        "api_key" : API_KEY,
        "format"  : "json"
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        return r.json().get("track", {})
    except:
        return {}


def kumpulkan_data() -> pd.DataFrame:
    all_records = []

    for genre in GENRES:
        print(f"\n[INFO] Mengambil genre: {genre.upper()}...")
        tracks = ambil_top_tracks(genre, TRACKS_PER_GENRE)

        for i, t in enumerate(tracks):
            artist_name = t.get("artist", {}).get("name", "")
            track_name  = t.get("name", "")
            rank        = t.get("@attr", {}).get("rank", "")   # ← rank di genre

            detail = ambil_detail_track(artist_name, track_name)

            # ── Tags ──
            tags_raw = detail.get("toptags", {}).get("tag", [])
            tags_str = ", ".join([tg["name"] for tg in tags_raw]) if tags_raw else ""

            # ── Album ──
            album_info  = detail.get("album", {})
            album_title = album_info.get("title", "")

            # ── Tahun rilis dari wiki.published ──
            wiki      = detail.get("wiki", {})
            published = wiki.get("published", "")
            # Format: "06 Jan 2017, 09:23" → ambil 4 karakter terakhir sebelum koma
            try:
                tahun_rilis = published.split(",")[0].strip().split()[-1]
            except:
                tahun_rilis = ""

            # ── Semua field original Last.fm ──
            record = {
                # ── Identitas ──
                "track_name"    : track_name,
                "artist"        : artist_name,
                "album"         : album_title,          # ← FIELD ORIGINAL
                "genre_tag"     : genre,
                "rank_in_genre" : rank,                 # ← FIELD ORIGINAL
                "tahun_rilis"   : tahun_rilis,          # ← dari wiki.published
                "mbid"          : detail.get("mbid",""),# ← FIELD ORIGINAL
                "url"           : detail.get("url",""), # ← FIELD ORIGINAL

                # ── Fitur Numerik Original ──
                "playcount"     : int(detail.get("playcount",  0) or 0),
                "listeners"     : int(detail.get("listeners",  0) or 0),
                "duration_ms"   : int(detail.get("duration",   0) or 0),
                "tag_count"     : len(tags_raw),
                "tags"          : tags_str,

                # ── Streamable (original) ──
                "streamable"    : detail.get("streamable", {}).get("#text", "0"),
            }
            all_records.append(record)

            if (i + 1) % 10 == 0:
                print(f"  Selesai {i+1}/{len(tracks)} lagu...")
            time.sleep(0.2)

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"\n[OK] Data RAW disimpan → {OUTPUT_RAW}  ({len(df)} baris)")
    return df


# =============================================================================
# 2. REKAYASA FITUR
# =============================================================================
def rekayasa_fitur(df: pd.DataFrame):
    print("\n" + "="*60)
    print("REKAYASA FITUR (FEATURE ENGINEERING)")
    print("="*60)

    df = df.copy()

    # ── Bersihkan data ──
    df.dropna(subset=["playcount", "listeners", "duration_ms"], inplace=True)
    df = df[(df["playcount"] > 0) & (df["listeners"] > 0) & (df["duration_ms"] > 0)]

    print(f"  Baris setelah cleaning : {len(df)}")

    # ────────────────────────────────────────────
    # FITUR BARU 1 — Konversi Durasi
    # Alasan: duration_ms susah dibaca, diubah ke menit dan detik
    # ────────────────────────────────────────────
    df["duration_min"] = df["duration_ms"] / 60000
    df["duration_sec"] = df["duration_ms"] / 1000

    # ────────────────────────────────────────────
    # FITUR BARU 2 — Transformasi Logaritmik
    # Alasan: playcount & listeners distribusinya sangat skewed (jutaan),
    # log transform membuat distribusi lebih normal → lebih baik untuk analisis
    # ────────────────────────────────────────────
    df["log_playcount"] = np.log1p(df["playcount"])
    df["log_listeners"] = np.log1p(df["listeners"])

    # ────────────────────────────────────────────
    # FITUR BARU 3 — Engagement Ratio
    # Alasan: mengukur loyalitas pendengar
    # (berapa kali rata-rata 1 orang memutar lagu ini)
    # ────────────────────────────────────────────
    df["engagement_ratio"] = df["playcount"] / (df["listeners"] + 1)

    # ────────────────────────────────────────────
    # FITUR BARU 4 — Avg Play per Listener
    # Alasan: mirip engagement ratio tapi menekankan
    # rata-rata pemutaran per orang, berguna sbg fitur independen
    # ────────────────────────────────────────────
    df["avg_play_per_listener"] = df["playcount"] / (df["listeners"] + 1)

    # ────────────────────────────────────────────
    # FITUR BARU 5 — Popularity Score (MinMax 0–1)
    # Alasan: normalisasi playcount agar bisa dibandingkan
    # lintas genre tanpa bias skala besar/kecil
    # ────────────────────────────────────────────
    scaler = MinMaxScaler()
    df["popularity_score"] = scaler.fit_transform(df[["playcount"]])

    # ────────────────────────────────────────────
    # FITUR BARU 6 — Kategori Durasi (Binning)
    # Alasan: durasi numerik dikelompokkan berdasarkan
    # domain knowledge industri musik
    # ────────────────────────────────────────────
    df["duration_category"] = pd.cut(
        df["duration_min"],
        bins  = [0,    2.5,           4.0,          6.0,              9999],
        labels= ["Pendek\n(<2.5m)", "Sedang\n(2.5-4m)",
                 "Panjang\n(4-6m)", "Sangat Panjang\n(>6m)"]
    )

    # ────────────────────────────────────────────
    # FITUR BARU 7 — Kelas Popularitas (Tertile)
    # Alasan: membagi lagu ke 3 kelas popularitas
    # berdasarkan distribusi data aktual (33%/66%)
    # ────────────────────────────────────────────
    q33 = df["playcount"].quantile(0.33)
    q66 = df["playcount"].quantile(0.66)
    df["popularity_class"] = pd.cut(
        df["playcount"],
        bins  = [-1, q33, q66, float("inf")],
        labels= ["Rendah", "Sedang", "Tinggi"]
    )

    # ────────────────────────────────────────────
    # FITUR BARU 8 — Label Encoding Genre
    # Alasan: genre berupa teks, diubah ke angka
    # agar bisa dipakai sebagai fitur numerik jika diperlukan
    # ────────────────────────────────────────────
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre_tag"])

    # Simpan dataset hasil rekayasa fitur
    df.to_csv(OUTPUT_FE, index=False)
    print(f"  Dataset hasil FE disimpan → {OUTPUT_FE}")

    # ── Tampilkan ringkasan fitur ──
    print("\n  === FITUR ORIGINAL (dari Last.fm) ===")
    original = ["track_name","artist","album","genre_tag","rank_in_genre",
                "tahun_rilis","mbid","url","playcount","listeners",
                "duration_ms","tag_count","tags","streamable"]
    for f in original:
        if f in df.columns:
            print(f"    ✔ {f}")

    print("\n  === FITUR HASIL REKAYASA ===")
    rekayasa = ["duration_min","duration_sec","log_playcount","log_listeners",
                "engagement_ratio","avg_play_per_listener","popularity_score",
                "duration_category","popularity_class","genre_encoded"]
    for f in rekayasa:
        print(f"    ✔ {f}")

    print(f"\n  Total kolom akhir : {len(df.columns)}")
    print(f"  Total baris bersih: {len(df)}")

    return df, le


# =============================================================================
# 3. STATISTIK DESKRIPTIF
# =============================================================================
def statistik_deskriptif(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STATISTIK DESKRIPTIF")
    print("="*60)

    fitur_num = ["playcount", "listeners", "duration_ms", "duration_min",
                 "avg_play_per_listener", "popularity_score",
                 "engagement_ratio", "tag_count"]

    desc        = df[fitur_num].describe().T
    desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
    print(desc.to_string())

    print("\n--- Rata-rata per Genre ---")
    print(df.groupby("genre_tag")[["playcount","listeners","duration_min",
                                    "engagement_ratio"]].mean().round(2).to_string())

    print("\n--- Distribusi Kategori Durasi ---")
    print(df["duration_category"].value_counts().to_string())

    print("\n--- Distribusi Kelas Popularitas ---")
    print(df["popularity_class"].value_counts().to_string())

    print("\n--- 5 Lagu dengan Playcount Tertinggi ---")
    print(df.nlargest(5, "playcount")[["track_name","artist","genre_tag",
                                        "playcount","listeners"]].to_string())

    print("\n--- 5 Lagu dengan Engagement Ratio Tertinggi ---")
    print(df.nlargest(5, "engagement_ratio")[["track_name","artist","genre_tag",
                                               "engagement_ratio","playcount"]].to_string())


# =============================================================================
# 4. VISUALISASI — LINE CHART & PIE CHART SAJA
# =============================================================================
def visualisasi(df: pd.DataFrame):
    print("\n" + "="*60)
    print("VISUALISASI DATA")
    print("="*60)

    palette = sns.color_palette("tab10", len(GENRES))
    genre_order = df.groupby("genre_tag")["playcount"].mean().sort_values(ascending=False).index.tolist()

    # ── 4.1 Line Chart — Rata-rata Playcount per Genre ──
    fig, ax = plt.subplots(figsize=(13, 5))
    means   = df.groupby("genre_tag")["playcount"].mean().reindex(genre_order)
    ax.plot(genre_order, means.values, marker="o", lw=2.5,
            color="#1565C0", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means.values):
        ax.annotate(f"{y/1e6:.2f}M", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#1565C0")
    ax.fill_between(range(len(genre_order)), means.values, alpha=0.08, color="#1565C0")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Rata-rata Playcount")
    ax.set_title("Rata-rata Playcount per Genre", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M"))
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_line_playcount_genre.png")
    plt.close()
    print("  [OK] 01_line_playcount_genre.png")

    # ── 4.2 Line Chart — Rata-rata Listeners per Genre ──
    fig, ax = plt.subplots(figsize=(13, 5))
    means_l = df.groupby("genre_tag")["listeners"].mean().reindex(genre_order)
    ax.plot(genre_order, means_l.values, marker="s", lw=2.5,
            color="#E65100", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means_l.values):
        ax.annotate(f"{y/1e6:.2f}M", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#E65100")
    ax.fill_between(range(len(genre_order)), means_l.values, alpha=0.08, color="#E65100")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Rata-rata Listeners")
    ax.set_title("Rata-rata Listeners per Genre", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M"))
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/02_line_listeners_genre.png")
    plt.close()
    print("  [OK] 02_line_listeners_genre.png")

    # ── 4.3 Line Chart — Rata-rata Durasi (menit) per Genre ──
    fig, ax = plt.subplots(figsize=(13, 5))
    means_d = df.groupby("genre_tag")["duration_min"].mean().reindex(genre_order)
    ax.plot(genre_order, means_d.values, marker="^", lw=2.5,
            color="#2E7D32", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means_d.values):
        ax.annotate(f"{y:.1f} mnt", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#2E7D32")
    ax.fill_between(range(len(genre_order)), means_d.values, alpha=0.08, color="#2E7D32")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Rata-rata Durasi (menit)")
    ax.set_title("Rata-rata Durasi Lagu per Genre", fontsize=14, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/03_line_durasi_genre.png")
    plt.close()
    print("  [OK] 03_line_durasi_genre.png")

    # ── 4.4 Line Chart — Engagement Ratio per Genre ──
    fig, ax = plt.subplots(figsize=(13, 5))
    means_e = df.groupby("genre_tag")["engagement_ratio"].mean().reindex(genre_order)
    ax.plot(genre_order, means_e.values, marker="D", lw=2.5,
            color="#6A1B9A", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means_e.values):
        ax.annotate(f"{y:.2f}x", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#6A1B9A")
    ax.fill_between(range(len(genre_order)), means_e.values, alpha=0.08, color="#6A1B9A")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Engagement Ratio")
    ax.set_title("Rata-rata Engagement Ratio per Genre\n(Berapa kali rata-rata 1 pendengar memutar ulang)", fontsize=13, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/04_line_engagement_genre.png")
    plt.close()
    print("  [OK] 04_line_engagement_genre.png")

    # ── 4.5 Line Chart — Multi-fitur (Normalized) semua genre ──
    fig, ax = plt.subplots(figsize=(13, 6))
    fitur_multi = {
        "Playcount (norm)" : "playcount",
        "Listeners (norm)" : "listeners",
        "Durasi (norm)"    : "duration_min",
        "Engagement (norm)": "engagement_ratio",
    }
    warna = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
    for (label, col), warna_ in zip(fitur_multi.items(), warna):
        vals = df.groupby("genre_tag")[col].mean().reindex(genre_order)
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        ax.plot(genre_order, vals_norm.values, marker="o", lw=2,
                label=label, color=warna_, ms=7)
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Nilai Ternormalisasi (0–1)")
    ax.set_title("Perbandingan Multi-Fitur Antar Genre (Ternormalisasi)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/05_line_multifitur_genre.png")
    plt.close()
    print("  [OK] 05_line_multifitur_genre.png")

    # ── 4.6 Pie Chart — Distribusi Kategori Durasi ──
    fig, ax = plt.subplots(figsize=(8, 8))
    dur_counts = df["duration_category"].value_counts()
    colors_pie = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]
    wedges, texts, autotexts = ax.pie(
        dur_counts,
        labels    = dur_counts.index,
        autopct   = "%1.1f%%",
        startangle= 140,
        colors    = colors_pie,
        pctdistance=0.80,
        wedgeprops = {"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Distribusi Kategori Durasi Lagu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/06_pie_kategori_durasi.png")
    plt.close()
    print("  [OK] 06_pie_kategori_durasi.png")

    # ── 4.7 Pie Chart — Distribusi Kelas Popularitas ──
    fig, ax = plt.subplots(figsize=(8, 8))
    pop_counts = df["popularity_class"].value_counts()
    colors_pop = ["#EF9A9A", "#FFCC80", "#A5D6A7"]
    wedges, texts, autotexts = ax.pie(
        pop_counts,
        labels    = pop_counts.index,
        autopct   = "%1.1f%%",
        startangle= 90,
        colors    = colors_pop,
        pctdistance=0.80,
        wedgeprops = {"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Distribusi Kelas Popularitas Lagu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/07_pie_kelas_popularitas.png")
    plt.close()
    print("  [OK] 07_pie_kelas_popularitas.png")

    # ── 4.8 Pie Chart — Distribusi Jumlah Lagu per Genre ──
    fig, ax = plt.subplots(figsize=(9, 9))
    genre_counts = df["genre_tag"].value_counts()
    colors_genre = sns.color_palette("tab10", len(genre_counts))
    wedges, texts, autotexts = ax.pie(
        genre_counts,
        labels    = genre_counts.index,
        autopct   = "%1.1f%%",
        startangle= 140,
        colors    = colors_genre,
        pctdistance=0.82,
        wedgeprops = {"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Distribusi Jumlah Lagu per Genre dalam Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/08_pie_distribusi_genre.png")
    plt.close()
    print("  [OK] 08_pie_distribusi_genre.png")

    print(f"\n  Total visualisasi: 8 chart disimpan di folder → {IMG_DIR}/")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("  REKAYASA FITUR — ANALISIS MUSIK LAST.FM")
    print("="*60)

    # Cek apakah CSV sudah ada (hemat API request)
    if os.path.exists(OUTPUT_RAW):
        print(f"\n[INFO] File CSV sudah ada, memuat dari: {OUTPUT_RAW}")
        df_raw = pd.read_csv(OUTPUT_RAW)
    else:
        print("\n[INFO] Mengambil data dari Last.fm API...")
        df_raw = kumpulkan_data()

    print(f"\n[INFO] Shape data mentah : {df_raw.shape}")
    print(f"[INFO] Kolom original    : {list(df_raw.columns)}")
    print(df_raw.head(3).to_string())

    # Pipeline
    df, le = rekayasa_fitur(df_raw)
    statistik_deskriptif(df)
    visualisasi(df)

    print("\n" + "="*60)
    print(f"  SELESAI!")
    print(f"  CSV Raw             : {OUTPUT_RAW}")
    print(f"  CSV Feature Eng.    : {OUTPUT_FE}")
    print(f"  Visualisasi         : folder {IMG_DIR}/")
    print("="*60)