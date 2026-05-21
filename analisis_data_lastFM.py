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

# KONFIGURASI
API_KEY    = "e89377b8d9c451a43ebc5bb42a9fe555"
BASE_URL   = "http://ws.audioscrobbler.com/2.0/"
OUTPUT_RAW = "PramesaAlyusufi_data_lastfmaudio_teks_API.csv"
OUTPUT_FE  = "PramesaAlyusufi_data_lastfmaudio_rekayasa_fitur.csv"
IMG_DIR    = "visualisasi"
os.makedirs(IMG_DIR, exist_ok=True)

GENRES           = ["pop", "rock", "jazz", "electronic", "hip-hop", "classical", "indie", "metal", "rnb", "country"]
TRACKS_PER_GENRE = 100

# PENGUMPULAN DATA
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
        print(f"Genre {genre}: {e}")
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
        print(f"\nMengambil genre: {genre.upper()}...")
        tracks = ambil_top_tracks(genre, TRACKS_PER_GENRE)

        for i, t in enumerate(tracks):
            artist_name = t.get("artist", {}).get("name", "")
            track_name  = t.get("name", "")
            rank        = t.get("@attr", {}).get("rank", "")

            detail = ambil_detail_track(artist_name, track_name)

            tags_raw = detail.get("toptags", {}).get("tag", [])
            tags_str = ", ".join([tg["name"] for tg in tags_raw]) if tags_raw else ""

            album_info  = detail.get("album", {})
            album_title = album_info.get("title", "")

            wiki      = detail.get("wiki", {})
            published = wiki.get("published", "")
            try:
                tahun_rilis = published.split(",")[0].strip().split()[-1]
            except:
                tahun_rilis = ""

            record = {
                "track_name"    : track_name,
                "artist"        : artist_name,
                "album"         : album_title,
                "genre_tag"     : genre,
                "rank_in_genre" : rank,
                "tahun_rilis"   : tahun_rilis,
                "mbid"          : detail.get("mbid",""),
                "url"           : detail.get("url",""),
                "playcount"     : int(detail.get("playcount",  0) or 0),
                "listeners"     : int(detail.get("listeners",  0) or 0),
                "duration_ms"   : int(detail.get("duration",   0) or 0),
                "tag_count"     : len(tags_raw),
                "tags"          : tags_str,
                "streamable"    : detail.get("streamable", {}).get("#text", "0"),
            }
            all_records.append(record)

            if (i + 1) % 10 == 0:
                print(f" Selesai {i+1}/{len(tracks)} lagu...")
            time.sleep(0.2)

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Data RAW disimpan => {OUTPUT_RAW}  ({len(df)} baris)")
    return df


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print("PRE-PROCESSING DATA")

    df = df.copy()
    awal = len(df)
    print(f" Jumlah data awal : {awal} baris")

    df.drop_duplicates(subset=["track_name", "artist"], inplace=True)
    print(f" Setelah drop duplikat: {len(df)} baris (hapus {awal - len(df)})")

    before = len(df)
    df.dropna(subset=["playcount", "listeners", "duration_ms", "genre_tag"], inplace=True)
    print(f" Setelah drop NaN : {len(df)} baris (hapus {before - len(df)})")

    before = len(df)
    df = df[(df["playcount"] > 0) &
            (df["listeners"] > 0) &
            (df["duration_ms"] > 0)]
    print(f" Setelah filter nol : {len(df)} baris (hapus {before - len(df)})")

    df["playcount"]   = pd.to_numeric(df["playcount"],   errors="coerce").fillna(0).astype(int)
    df["listeners"]   = pd.to_numeric(df["listeners"],   errors="coerce").fillna(0).astype(int)
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce").fillna(0).astype(int)
    df["tag_count"]   = pd.to_numeric(df["tag_count"],   errors="coerce").fillna(0).astype(int)
    print(f" Tipe data diperbaiki : playcount, listeners, duration_ms => int")

    df["track_name"] = df["track_name"].astype(str).str.strip()
    df["artist"]     = df["artist"].astype(str).str.strip()
    df["genre_tag"]  = df["genre_tag"].astype(str).str.strip().str.lower()
    print(f" Teks dibersihkan     : strip spasi, genre => lowercase")

    Q1  = df["playcount"].quantile(0.25)
    Q3  = df["playcount"].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df)
    df = df[df["playcount"].between(Q1 - 3*IQR, Q3 + 3*IQR)]
    print(f" Setelah hapus outlier: {len(df)} baris (hapus {before - len(df)})")

    print(f"\n Total baris bersih : {len(df)} (dari {awal} awal)")
    print(f" Data hilang total : {awal - len(df)} baris ({(awal-len(df))/awal*100:.1f}%)")

    return df


# REKAYASA FITUR
def rekayasa_fitur(df: pd.DataFrame):
    print("REKAYASA FITUR")
    df = df.copy()

    df.dropna(subset=["playcount", "listeners", "duration_ms"], inplace=True)
    df = df[(df["playcount"] > 0) & (df["listeners"] > 0) & (df["duration_ms"] > 0)]

    print(f" Baris setelah cleaning : {len(df)}")

    df["duration_min"] = df["duration_ms"] / 60000
    df["duration_sec"] = df["duration_ms"] / 1000

    df["log_playcount"] = np.log1p(df["playcount"])
    df["log_listeners"] = np.log1p(df["listeners"])

    df["engagement_ratio"] = df["playcount"] / (df["listeners"] + 1)

    df["avg_play_per_listener"] = df["playcount"] / (df["listeners"] + 1)

    scaler = MinMaxScaler()
    df["popularity_score"] = scaler.fit_transform(df[["playcount"]])

    df["duration_category"] = pd.cut(
        df["duration_min"],
        bins  = [0, 2.5, 4.0, 6.0, 9999],
        labels= ["Pendek\n(<2.5m)", "Sedang\n(2.5-4m)", "Panjang\n(4-6m)", "Sangat Panjang\n(>6m)"]
    )

    q33 = df["playcount"].quantile(0.33)
    q66 = df["playcount"].quantile(0.66)
    df["popularity_class"] = pd.cut(
        df["playcount"],
        bins  = [-1, q33, q66, float("inf")],
        labels= ["Rendah", "Sedang", "Tinggi"]
    )

    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre_tag"])

    df.to_csv(OUTPUT_FE, index=False)
    print(f"Dataset hasil FE disimpan = {OUTPUT_FE}")

    print("\n FITUR DATA LAST.FM")
    original = ["track_name","artist","album","genre_tag","rank_in_genre", "tahun_rilis","mbid","url","playcount","listeners", "duration_ms","tag_count","tags","streamable"]
    for f in original:
        if f in df.columns:
            print(f"Selesai {f}")

    print("\n FITUR HASIL DARI REKAYASA FEATURE")
    rekayasa = ["duration_min","duration_sec","log_playcount","log_listeners", "engagement_ratio","avg_play_per_listener","popularity_score", "duration_category","popularity_class","genre_encoded"]
    for f in rekayasa:
        print(f"Selesai {f}")

    print(f"\n Total kolom akhir : {len(df.columns)}")
    print(f"Total baris bersih: {len(df)}")

    return df, le


# STATISTIK DESKRIPTIF
def statistik_deskriptif(df: pd.DataFrame):
    print("STATISTIK DESKRIPTIF")

    fitur_num = ["playcount", "listeners", "duration_ms", "duration_min", "avg_play_per_listener", "popularity_score", "engagement_ratio", "tag_count"]

    desc = df[fitur_num].describe().T
    desc["median"] = df[fitur_num].median()
    desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
    print(desc.to_string())

    print("\n Rata rata per Genre ")
    print(df.groupby("genre_tag")[["playcount","listeners","duration_min", "engagement_ratio"]].mean().round(2).to_string())

    print("\n Distribusi Kategori Durasi ")
    print(df["duration_category"].value_counts().to_string())

    print("\n Distribusi Kelas Popularitas ")
    print(df["popularity_class"].value_counts().to_string())

    print("\n 5 Lagu dengan Playcount Tertinggi ")
    print(df.nlargest(5, "playcount")[["track_name","artist","genre_tag", "playcount","listeners"]].to_string())

    print("\n 5 Lagu dengan Engagement Ratio Tertinggi ")
    print(df.nlargest(5, "engagement_ratio")[["track_name","artist","genre_tag", "engagement_ratio","playcount"]].to_string())


# VISUALISASI — LINE CHART & PIE CHART
def visualisasi(df: pd.DataFrame):
    print("VISUALISASI DATA")

    palette = sns.color_palette("tab10", len(GENRES))
    genre_order = df.groupby("genre_tag")["playcount"].mean().sort_values(ascending=False).index.tolist()

    # Line Chart — Mean Playcount per Genre
    fig, ax = plt.subplots(figsize=(13, 5))
    means   = df.groupby("genre_tag")["playcount"].mean().reindex(genre_order)
    ax.plot(genre_order, means.values, marker="o", lw=2.5, color="#1565C0", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means.values):
        ax.annotate(f"{y/1e6:.2f}M", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#1565C0")
    ax.fill_between(range(len(genre_order)), means.values, alpha=0.08, color="#1565C0")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Rata rata Playcount")
    ax.set_title("Rata rata Playcount per Genre", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M"))
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_line_playcount_genre.png")
    plt.close()
    print("01_line_playcount_genre.png")

    # Line Chart — Mean Listeners per Genre
    fig, ax = plt.subplots(figsize=(13, 5))
    means_l = df.groupby("genre_tag")["listeners"].mean().reindex(genre_order)
    ax.plot(genre_order, means_l.values, marker="s", lw=2.5, color="#E65100", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means_l.values):
        ax.annotate(f"{y/1e6:.2f}M", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#E65100")
    ax.fill_between(range(len(genre_order)), means_l.values, alpha=0.08, color="#E65100")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Rata rata Listeners")
    ax.set_title("Rata rata Listeners per Genre", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M"))
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/02_line_listeners_genre.png")
    plt.close()
    print("02_line_listeners_genre.png")

    # Line Chart — Rata rata Durasi per Genre
    fig, ax = plt.subplots(figsize=(13, 5))
    means_d = df.groupby("genre_tag")["duration_min"].mean().reindex(genre_order)
    ax.plot(genre_order, means_d.values, marker="^", lw=2.5, color="#2E7D32", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means_d.values):
        ax.annotate(f"{y:.1f} mnt", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#2E7D32")
    ax.fill_between(range(len(genre_order)), means_d.values, alpha=0.08, color="#2E7D32")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Rata rata Durasi")
    ax.set_title("Rata rata Durasi Lagu per Genre", fontsize=14, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/03_line_durasi_genre.png")
    plt.close()
    print("03_line_durasi_genre.png")

    # Line Chart — Engagement Ratio per Genre
    fig, ax = plt.subplots(figsize=(13, 5))
    means_e = df.groupby("genre_tag")["engagement_ratio"].mean().reindex(genre_order)
    ax.plot(genre_order, means_e.values, marker="D", lw=2.5,
            color="#6A1B9A", ms=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y in enumerate(means_e.values):
        ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#6A1B9A")
    ax.fill_between(range(len(genre_order)), means_e.values, alpha=0.08, color="#6A1B9A")
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Engagement Ratio")
    ax.set_title("Rata rata Engagement Ratio per Genre\n(Berapa kali rata rata 1 pendengar memutar ulang)", fontsize=13, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/04_line_engagement_genre.png")
    plt.close()
    print("04_line_engagement_genre.png")

    # Line Chart — Multi-fitur (Normalized) semua genre
    fig, ax = plt.subplots(figsize=(13, 6))
    fitur_multi = {
        "Playcount_norm" : "playcount",
        "Listeners_norm" : "listeners",
        "Durasi_norm"    : "duration_min",
        "Engagement_norm": "engagement_ratio",
    }

    warna = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
    for (label, col), warna_ in zip(fitur_multi.items(), warna):
        vals = df.groupby("genre_tag")[col].mean().reindex(genre_order)
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        ax.plot(
            genre_order, 
            vals_norm.values, 
            marker="o", 
            lw=2,label=label, 
            color=warna_, 
            ms=7
            )
    ax.set_xticks(range(len(genre_order)))
    ax.set_xticklabels(genre_order, rotation=20, ha="right")
    ax.set_ylabel("Nilai Ternormalisasi (0–1)")
    ax.set_title("Perbandingan Multi-Fitur Antar Genre (Ternormalisasi)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/05_line_multifitur_genre.png")
    plt.close()
    print("05_line_multifitur_genre.png")

    # Pie Chart — Distribusi Kategori Durasi
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
    print("06_pie_kategori_durasi.png")

    # Pie Chart — Distribusi Kelas Popularitas
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
    print("07_pie_kelas_popularitas.png")

    # Pie Chart — Distribusi Jumlah Lagu per Genre
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
    print("08_pie_distribusi_genre.png")

    print(f"\n Visualisasi chart disimpan di folder => {IMG_DIR}/")

# MAIN
if __name__ == "__main__":
    print("ANALISIS MUSIK LAST.FM")

    if os.path.exists(OUTPUT_RAW):
        print(f"\nFile CSV sudah ada, memuat dari {OUTPUT_RAW}")
        df_raw = pd.read_csv(OUTPUT_RAW)
    else:
        print("\n Mengambil data dari Last.fm API...")
        df_raw = kumpulkan_data()

    print(f"\nShape data mentah : {df_raw.shape}")
    print(f"Kolom original    : {list(df_raw.columns)}")
    print(df_raw.head(3).to_string())

    df, le = rekayasa_fitur(df_raw)
    statistik_deskriptif(df)
    visualisasi(df)

    print(f" SELESAI!")
    print(f" CSV Raw : {OUTPUT_RAW}")
    print(f" CSV Rekayasa Fitur : {OUTPUT_FE}")
    print(f" Visualisasi : folder {IMG_DIR}/")
