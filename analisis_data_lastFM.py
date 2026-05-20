import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

# 0. KONFIGURASI
API_KEY   = "e89377b8d9c451a43ebc5bb42a9fe555"
BASE_URL  = "http://ws.audioscrobbler.com/2.0/"
OUTPUT_CSV = "NamaLengkap_data_lastfmaudio_teks_API.csv"
IMG_DIR    = "visualisasi"
os.makedirs(IMG_DIR, exist_ok=True)

GENRES = ["pop", "rock", "jazz", "electronic", "hip-hop", "classical", "indie", "metal", "rnb", "country"]
TRACKS_PER_GENRE = 100   # Ubah ke 100 untuk lebih banyak data

# 1. PENGUMPULAN DATA - LAST.FM API
def ambil_top_tracks(genre: str, limit: int = 50) -> list:
    """Ambil top tracks berdasarkan tag/genre dari Last.fm."""
    params = {
        "method"  : "tag.gettoptracks",
        "tag"     : genre,
        "api_key" : API_KEY,
        "format"  : "json",
        "limit"   : limit
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        data = r.json()
        return data.get("tracks", {}).get("track", [])
    except Exception as e:
        print(f"  [ERROR] Genre {genre}: {e}")
        return []


def ambil_detail_track(artist: str, track: str) -> dict:
    """Ambil detail (playcount, listeners, durasi, tags) dari 1 lagu."""
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
    """Pipeline pengumpulan data dari Last.fm untuk semua genre."""
    all_records = []

    for genre in GENRES:
        print(f"\n[INFO] Mengambil genre: {genre.upper()}...")
        tracks = ambil_top_tracks(genre, TRACKS_PER_GENRE)

        for i, t in enumerate(tracks):
            artist_name = t.get("artist", {}).get("name", "")
            track_name  = t.get("name", "")

            detail = ambil_detail_track(artist_name, track_name)

            tags_raw = detail.get("toptags", {}).get("tag", [])
            tags_str = ", ".join([tg["name"] for tg in tags_raw]) if tags_raw else ""

            record = {
                "track_name"  : track_name,
                "artist"      : artist_name,
                "genre_tag"   : genre,
                "playcount"   : int(detail.get("playcount", 0) or 0),
                "listeners"   : int(detail.get("listeners", 0) or 0),
                "duration_ms" : int(detail.get("duration", 0) or 0),
                "tags"        : tags_str,
                "tag_count"   : len(tags_raw),
            }
            all_records.append(record)

            if (i + 1) % 10 == 0:
                print(f"  Selesai {i+1}/{len(tracks)} lagu...")
            time.sleep(0.2)

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Data disimpan → {OUTPUT_CSV}  ({len(df)} baris)")
    return df


# 2. SIMULASI DATA (dipakai jika API Key belum diisi / untuk testing)
def buat_data_simulasi() -> pd.DataFrame:
    """Buat data simulasi realistis jika API key belum tersedia."""
    np.random.seed(42)
    genres = ["pop","rock","jazz","electronic","hip-hop", "classical","indie","metal","rnb","country"]
    n = 80

    rows = []
    for g in genres:
        base_play = {"pop":5e6,"rock":3e6,"jazz":1e6,"electronic":2.5e6, "hip-hop":4e6,"classical":800e3,"indie":1.5e6, "metal":2e6,"rnb":3.5e6,"country":2e6}[g]
        base_list = base_play * np.random.uniform(0.5, 0.9)
        base_dur  = {"pop":210,"rock":250,"jazz":320,"electronic":280, "hip-hop":200,"classical":400,"indie":230,"metal":290, "rnb":220,"country":240}[g]

        for i in range(n):
            play = int(abs(np.random.normal(base_play, base_play * 0.3)))
            lst  = int(abs(np.random.normal(base_list, base_list * 0.3)))
            dur  = int(abs(np.random.normal(base_dur, 40)) * 1000)
            rows.append({
                "track_name" : f"{g.title()} Track {i+1}",
                "artist"     : f"Artist_{g}_{i+1}",
                "genre_tag"  : g,
                "playcount"  : play,
                "listeners"  : lst,
                "duration_ms": dur,
                "tags"       : f"{g}, music",
                "tag_count"  : np.random.randint(2, 8),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Data simulasi dibuat → {OUTPUT_CSV}  ({len(df)} baris)")
    return df


# 3. REKAYASA FITUR (FEATURE ENGINEERING)
def rekayasa_fitur(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("REKAYASA FITUR")
    print("="*60)

    df = df.copy()
    df.dropna(subset=["playcount","listeners","duration_ms"], inplace=True)
    df = df[(df["playcount"] > 0) & (df["listeners"] > 0)]

    # --- Fitur baru ---
    df["avg_play_per_listener"] = df["playcount"] / (df["listeners"] + 1)
    df["duration_min"]          = df["duration_ms"] / 60000
    df["duration_sec"]          = df["duration_ms"] / 1000

    # Kategori durasi
    df["duration_category"] = pd.cut(
        df["duration_min"],
        bins=[0, 2.5, 4.0, 6.0, 9999],
        labels=["Pendek (<2.5m)", "Sedang (2.5-4m)", "Panjang (4-6m)", "Sangat Panjang (>6m)"]
    )

    # Skor popularitas (Min-Max 0-1)
    df["popularity_score"] = (df["playcount"] - df["playcount"].min()) / \
                              (df["playcount"].max() - df["playcount"].min() + 1e-9)

    # Log transform (mengurangi skewness)
    df["log_playcount"]  = np.log1p(df["playcount"])
    df["log_listeners"]  = np.log1p(df["listeners"])

    # Engagement ratio
    df["engagement_ratio"] = np.where(
        df["listeners"] > 0,
        df["playcount"] / df["listeners"],
        0
    )

    # Kategori popularitas berdasarkan playcount
    q33 = df["playcount"].quantile(0.33)
    q66 = df["playcount"].quantile(0.66)
    df["popularity_class"] = pd.cut(
        df["playcount"],
        bins=[-1, q33, q66, float("inf")],
        labels=["Rendah", "Sedang", "Tinggi"]
    )

    # Encoding genre → angka
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre_tag"])

    print(f"Jumlah baris bersih : {len(df)}")
    print(f"Fitur baru dibuat : avg_play_per_listener, duration_min, popularity_score,")
    print(f"log_playcount, log_listeners, engagement_ratio,")
    print(f"duration_category, popularity_class, genre_encoded")

    return df, le


# 4. STATISTIK DESKRIPTIF
def statistik_deskriptif(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STATISTIK DESKRIPTIF")
    print("="*60)

    fitur_num = ["playcount","listeners","duration_ms","duration_min", "avg_play_per_listener","popularity_score","engagement_ratio","tag_count"]
    desc = df[fitur_num].describe().T
    desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
    print(desc.to_string())

    print("\n--- Distribusi per Genre ---")
    print(df.groupby("genre_tag")[["playcount","listeners"]].mean().round(0).to_string())

    print("\n--- Distribusi Kategori Durasi ---")
    print(df["duration_category"].value_counts().to_string())

    print("\n--- Distribusi Kategori Popularitas ---")
    print(df["popularity_class"].value_counts().to_string())


# 5. VISUALISASI
def visualisasi(df: pd.DataFrame):
    print("\n" + "="*60)
    print("VISUALISASI DATA")
    print("="*60)
    genres = df["genre_tag"].unique()
    palette = sns.color_palette("tab10", len(genres))

    # --- 5.1 Distribusi Playcount per Genre (KDE) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, g in enumerate(genres):
        subset = df[df["genre_tag"] == g]["log_playcount"]
        subset.plot.kde(ax=ax, label=g, color=palette[i], lw=2)
    ax.set_title("Distribusi Log Playcount per Genre", fontsize=14, fontweight="bold")
    ax.set_xlabel("Log(Playcount + 1)")
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_distribusi_playcount.png")
    plt.close()
    print("  [OK] 01_distribusi_playcount.png")

    # --- 5.2 Boxplot Playcount per Genre ---
    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column="playcount", by="genre_tag", ax=ax, patch_artist=True, notch=False)
    ax.set_title("Perbandingan Playcount per Genre", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Playcount")
    plt.suptitle("")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/02_boxplot_playcount.png")
    plt.close()
    print("  [OK] 02_boxplot_playcount.png")

    # --- 5.3 Heatmap Korelasi ---
    fitur_corr = ["playcount","listeners","duration_min", "avg_play_per_listener","popularity_score", "engagement_ratio","tag_count","log_playcount"]
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[fitur_corr].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Heatmap Korelasi Antar Fitur", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/03_heatmap_korelasi.png")
    plt.close()
    print("  [OK] 03_heatmap_korelasi.png")

    # --- 5.4 Scatter Playcount vs Listeners ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, g in enumerate(genres):
        sub = df[df["genre_tag"] == g]
        ax.scatter(sub["listeners"], sub["playcount"], label=g, alpha=0.6, s=50, color=palette[i])
    ax.set_xlabel("Listeners")
    ax.set_ylabel("Playcount")
    ax.set_title("Hubungan Listeners vs Playcount per Genre", fontsize=14, fontweight="bold")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/04_scatter_play_listen.png")
    plt.close()
    print("  [OK] 04_scatter_play_listen.png")

    # --- 5.5 Bar Chart Rata-rata Fitur per Genre ---
    genre_means = df.groupby("genre_tag")[["playcount","listeners","duration_min","engagement_ratio"]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    cols = ["playcount","listeners","duration_min","engagement_ratio"]
    titles = ["Rata-rata Playcount","Rata-rata Listeners", "Rata-rata Durasi (menit)","Rata-rata Engagement Ratio"]
    for ax, col, title in zip(axes.flat, cols, titles):
        genre_means[col].sort_values().plot.barh(ax=ax, color=palette[:len(genre_means)])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(col)
    plt.suptitle("Perbandingan Rata-rata Fitur Antar Genre", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/05_barchart_genre.png")
    plt.close()
    print("  [OK] 05_barchart_genre.png")

    # --- 5.6 Pie Chart Distribusi Kategori Durasi ---
    fig, ax = plt.subplots(figsize=(7, 7))
    dur_counts = df["duration_category"].value_counts()
    ax.pie(dur_counts, labels=dur_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
    ax.set_title("Distribusi Kategori Durasi Lagu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/06_pie_durasi.png")
    plt.close()
    print("  [OK] 06_pie_durasi.png")

    # --- 5.7 Radar Chart Profil Genre ---
    cats = ["playcount","listeners","duration_min","engagement_ratio","tag_count"]
    gm = df.groupby("genre_tag")[cats].mean()
    # Normalisasi untuk radar
    gm_norm = (gm - gm.min()) / (gm.max() - gm.min() + 1e-9)

    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (genre, row) in enumerate(gm_norm.iterrows()):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, label=genre, lw=2, color=palette[i])
        ax.fill(angles, vals, alpha=0.08, color=palette[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("Radar Chart Profil Fitur per Genre\n(Ternormalisasi)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/07_radar_genre.png")
    plt.close()
    print("  [OK] 07_radar_genre.png")

    return corr


# 6. K-MEANS CLUSTERING
def kmeans_clustering(df: pd.DataFrame):
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING")
    print("="*60)

    fitur = ["log_playcount","log_listeners","duration_min", "avg_play_per_listener","engagement_ratio"]
    X = df[fitur].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow method
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_range, inertias, "bo-", lw=2, ms=8)
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Pemilihan K Optimal", fontweight="bold")
    ax.axvline(x=3, color="red", linestyle="--", label="K optimal = 3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/08_elbow_kmeans.png")
    plt.close()
    print("  [OK] 08_elbow_kmeans.png")

    # Fit dengan K=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    print("\n--- Rata-rata Fitur per Cluster ---")
    cluster_stats = df.groupby("cluster")[["playcount","listeners","duration_min","engagement_ratio"]].mean()
    print(cluster_stats.round(2).to_string())

    # Label cluster otomatis berdasarkan playcount
    rank = cluster_stats["playcount"].rank()
    label_map = {rank.idxmin(): "Niche (Rendah)", rank.idxmax(): "Viral (Tinggi)"}
    mid = [c for c in [0,1,2] if c not in label_map][0]
    label_map[mid] = "Mainstream (Sedang)"
    df["cluster_label"] = df["cluster"].map(label_map)
    print("\n--- Label Cluster ---")
    print(df["cluster_label"].value_counts().to_string())

    # Scatter cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Niche (Rendah)": "#2196F3", "Mainstream (Sedang)": "#FF9800", "Viral (Tinggi)": "#F44336"}
    for lbl, clr in colors.items():
        sub = df[df["cluster_label"] == lbl]
        ax.scatter(sub["log_listeners"], sub["log_playcount"], label=lbl, color=clr, alpha=0.7, s=60, edgecolors="white", lw=0.5)
    ax.set_xlabel("Log(Listeners)")
    ax.set_ylabel("Log(Playcount)")
    ax.set_title("K-Means Clustering Lagu (K=3)", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/09_scatter_cluster.png")
    plt.close()
    print("  [OK] 09_scatter_cluster.png")

    return df


# 7. PCA
def pca_analisis(df: pd.DataFrame):
    print("\n" + "="*60)
    print("PCA (PRINCIPAL COMPONENT ANALYSIS)")
    print("="*60)

    fitur = ["log_playcount","log_listeners","duration_min", "avg_play_per_listener","engagement_ratio","tag_count","popularity_score"]
    X = df[fitur].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # Explained variance semua komponen
    pca_full = PCA()
    pca_full.fit(X_scaled)
    var_ratio = pca_full.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(var_ratio)+1), var_ratio * 100, color="#4CAF50", alpha=0.8)
    ax.plot(range(1, len(var_ratio)+1), np.cumsum(var_ratio) * 100, "ro-", lw=2, ms=7, label="Kumulatif")
    ax.axhline(y=80, color="gray", linestyle="--", label="80% threshold")
    ax.set_xlabel("Komponen Utama")
    ax.set_ylabel("Variansi (%)")
    ax.set_title("Explained Variance PCA", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/10_pca_variance.png")
    plt.close()
    print("  [OK] 10_pca_variance.png")

    # PCA 2D visualisasi
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_scaled)
    print(f"\n  PC1 menjelaskan: {var_ratio[0]*100:.1f}%")
    print(f"  PC2 menjelaskan: {var_ratio[1]*100:.1f}%")
    print(f"  Total 2 PC    : {sum(var_ratio[:2])*100:.1f}%")

    genres = df["genre_tag"].unique()
    palette = sns.color_palette("tab10", len(genres))

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, g in enumerate(genres):
        idx = df["genre_tag"] == g
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=g, color=palette[i], alpha=0.7, s=60, edgecolors="white", lw=0.5)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title("PCA 2D — Visualisasi Lagu berdasarkan Genre", fontsize=14, fontweight="bold")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/11_pca_2d_genre.png")
    plt.close()
    print("  [OK] 11_pca_2d_genre.png")

    return pca2, var_ratio


# 8. RANDOM FOREST KLASIFIKASI GENRE
def random_forest_klasifikasi(df: pd.DataFrame, le: LabelEncoder):
    print("\n" + "="*60)
    print("RANDOM FOREST — KLASIFIKASI GENRE")
    print("="*60)

    fitur = ["playcount","listeners","duration_min","avg_play_per_listener", "popularity_score","engagement_ratio","tag_count","log_playcount","log_listeners"]
    X = df[fitur].fillna(0)
    y = df["genre_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", min_samples_leaf=2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Akurasi Model : {acc*100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature Importance
    imp = pd.Series(rf.feature_importances_, index=fitur).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1565C0" if v >= imp.median() else "#90CAF9" for v in imp]
    imp.plot.barh(ax=ax, color=colors)
    ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.axvline(x=imp.median(), color="red", linestyle="--", label="Median")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/12_feature_importance.png")
    plt.close()
    print("  [OK] 12_feature_importance.png")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix — Klasifikasi Genre", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/13_confusion_matrix.png")
    plt.close()
    print("  [OK] 13_confusion_matrix.png")

    return rf, imp, acc


# MAIN
if __name__ == "__main__":
    print("="*60)
    print("  REKAYASA FITUR — ANALISIS MUSIK LAST.FM")
    print("="*60)

    # --- Ambil data ---
    if API_KEY == "e89377b8d9c451a43ebc5bb42a9fe555":
        print("\n[PERINGATAN] API key belum diisi. Menggunakan DATA SIMULASI.")
        print("Ganti nilai API_KEY di baris 27 dengan API key Last.fm kamu!\n")
        df_raw = buat_data_simulasi()
    else:
        if os.path.exists(OUTPUT_CSV):
            print(f"[INFO] Memuat data dari file: {OUTPUT_CSV}")
            df_raw = pd.read_csv(OUTPUT_CSV)
        else:
            df_raw = kumpulkan_data()

    print(f"\n[INFO] Shape data mentah: {df_raw.shape}")
    print(df_raw.head(3).to_string())

    # --- Pipeline ---
    df, le        = rekayasa_fitur(df_raw)
    statistik_deskriptif(df)
    corr          = visualisasi(df)
    df            = kmeans_clustering(df)
    pca2, var_rat = pca_analisis(df)
    rf, imp, acc  = random_forest_klasifikasi(df, le)

    print("\n" + "="*60)
    print("Visualisasi Selesai dan tersimpan di folder : ", IMG_DIR)
    print("File CSV tersimpan di :", OUTPUT_CSV)
    print("="*60)
