import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPK Diabetes — AHP+TOPSIS", page_icon="🩺", layout="wide")

# ── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model_spk_diabetes.joblib")

model = load_model()
weights = model["weights"]
criteria = model["criteria"]
criteria_type = model["criteria_type"]
cr = model["consistency_ratio"]

criteria_short = ["Efektivitas", "Biaya", "Kemudahan", "Efek Samping", "Kecepatan"]
criteria_unit = ["%", "Rp rb/bln", "1-10", "1-10", "bln"]
criteria_desc = [
    "Persentase keberhasilan penatalaksanaan",
    "Estimasi biaya per bulan (ribu rupiah)",
    "Skor kemudahan penerapan (1 = sulit, 10 = mudah)",
    "Skor potensi efek samping (1 = ringan, 10 = berat)",
    "Estimasi waktu sampai hasil terlihat (bulan)",
]

# ── 5 Alternatif Referensi (nilai dari notebook) ────────────
alt_names = [
    "A1: Diet Rendah Karb + Olahraga",
    "A2: Konsultasi Dokter + HbA1c",
    "A3: Metformin + Monitor Gula Darah",
    "A4: Program Penurunan Berat Badan",
    "A5: Intervensi Gaya Hidup Intensif",
]
alt_short = ["Diet+Olahraga", "Konsultasi Dokter", "Metformin", "Penurunan BB", "Gaya Hidup Intensif"]
ref_matrix = np.array([
    [80, 150, 8, 2, 3],
    [75, 200, 6, 1, 1],
    [90, 350, 7, 5, 1],
    [70, 100, 9, 1, 4],
    [88, 500, 5, 2, 2],
], dtype=float)


# ── Fungsi TOPSIS ───────────────────────────────────────────
def topsis(decision_matrix, weights, criteria_type):
    n = decision_matrix.shape[1]
    col_norms = np.sqrt((decision_matrix**2).sum(axis=0))
    R = decision_matrix / col_norms
    V = R * weights

    A_plus = np.zeros(n)
    A_minus = np.zeros(n)
    for j, ct in enumerate(criteria_type):
        if ct == "benefit":
            A_plus[j] = V[:, j].max()
            A_minus[j] = V[:, j].min()
        else:
            A_plus[j] = V[:, j].min()
            A_minus[j] = V[:, j].max()

    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    scores = D_minus / (D_plus + D_minus)
    return scores


# ── Tampilan Jurnal / Referensi (hanya sekali, pakai session_state) ──
if "show_jurnal" not in st.session_state:
    st.session_state.show_jurnal = False
if "show_ref" not in st.session_state:
    st.session_state.show_ref = False

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.title("⚙️ Profil Pasien")
st.sidebar.markdown(
    "<small>Masukkan nilai untuk setiap kriteria sesuai kondisi pasien.</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

user_values = []
for i in range(5):
    ctype = criteria_type[i]
    label = f"{criteria_short[i]} ({criteria_unit[i]})"
    help_text = f"{criteria_desc[i]}  |  Tipe: **{ctype.upper()}**"
    if i == 0:  # Efektivitas %
        val = st.sidebar.slider(label, 10, 100, 78, help=help_text)
    elif i == 1:  # Biaya
        val = st.sidebar.slider(label, 50, 600, 250, 10, help=help_text)
    elif i == 2:  # Kemudahan
        val = st.sidebar.slider(label, 1, 10, 7, help=help_text)
    elif i == 3:  # Efek Samping
        val = st.sidebar.slider(label, 1, 10, 3, help=help_text)
    else:  # Kecepatan
        val = st.sidebar.slider(label, 1, 12, 3, help=help_text)
    user_values.append(float(val))

st.sidebar.markdown("---")

# Tombol referensi jurnal
if st.sidebar.button("📚 Lihat Referensi Jurnal"):
    st.session_state.show_jurnal = not st.session_state.show_jurnal

# ── Main Content ────────────────────────────────────────────
st.title("🩺 SPK Diabetes Melitus Tipe 2")
st.markdown("**Sistem Pendukung Keputusan — Metode AHP + TOPSIS**")
st.caption("Rekomendasi penatalaksanaan berdasarkan 5 kriteria klinis")

# ── Tab 1: Hasil ────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Hasil & Rekomendasi", "📋 Data Alternatif", "ℹ️ Metodologi"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    # ── Jalankan TOPSIS ──
    user_row = np.array([user_values], dtype=float)
    full_matrix = np.vstack([ref_matrix, user_row])
    scores = topsis(full_matrix, weights, criteria_type)

    # Pisahkan skor referensi dan user
    ref_scores = scores[:5]
    user_score = scores[5]
    all_scores = np.concatenate([ref_scores, [user_score]])
    rankings = pd.Series(all_scores).rank(ascending=False).astype(int).values
    user_rank = rankings[5]

    # Dataframe hasil
    df_result = pd.DataFrame(
        {
            "Alternatif": alt_names + ["🟢 **Profil Anda**"],
            "Skor TOPSIS": np.round(all_scores, 5),
            "Ranking": rankings,
        }
    ).sort_values("Ranking")
    df_result_display = df_result.copy()
    df_result_display["Ranking"] = df_result_display["Ranking"].apply(
        lambda r: ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][r - 1] + f" Rank {r}"
    )

    with col1:
        st.subheader("🏆 Hasil Perankingan")
        # Highlight user row
        st.dataframe(
            df_result_display.set_index("Alternatif"),
            use_container_width=True,
            column_config={
                "Skor TOPSIS": st.column_config.NumberColumn(format="%.5f"),
            },
        )

        # Rekomendasi terbaik
        best_idx = np.argmax(ref_scores)
        st.success(
            f"**Rekomendasi terbaik untuk profil ini:**\n\n"
            f"👉 {alt_names[best_idx]}"
        )
        if user_rank == 1:
            st.balloons()
            st.info("Nilai profil Anda paling optimal! 🎉")

    with col2:
        st.subheader("📈 Visualisasi Skor")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [
            "#2E74B5", "#4BACC6", "#A9D18E", "#FFD966", "#F4B183", "#E74C3C"
        ]
        all_labels = alt_short + ["Profil Anda"]
        sorted_idx = np.argsort(all_scores)[::-1]
        sorted_scores = all_scores[sorted_idx]
        sorted_labels = [all_labels[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]

        bars = ax.barh(range(6), sorted_scores, color=sorted_colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(6))
        ax.set_yticklabels(sorted_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Skor TOPSIS")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, sc in zip(bars, sorted_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{sc:.4f}", va="center", fontsize=9)
        ax.set_title("Skor TOPSIS — 5 Alternatif + Profil Anda", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    # ── Tabel perbandingan nilai ──
    st.markdown("---")
    st.subheader("📋 Perbandingan Nilai Kriteria")
    comp_df = pd.DataFrame(
        np.vstack([ref_matrix, user_row]),
        index=alt_short + ["**Profil Anda**"],
        columns=criteria_short,
    )
    for j, ct in enumerate(criteria_type):
        comp_df[f"{criteria_short[j]}"] = comp_df[f"{criteria_short[j]}"].astype(int)
    st.dataframe(comp_df, use_container_width=True)


with tab2:
    st.subheader("📋 5 Alternatif Penatalaksanaan Diabetes")
    df_alt = pd.DataFrame(
        ref_matrix.astype(int),
        index=alt_short,
        columns=criteria_short,
    )
    st.dataframe(df_alt, use_container_width=True)

    if st.button("📊 Lihat Data Tambahan", key="show_ref_btn"):
        st.session_state.show_ref = not st.session_state.show_ref

    if st.session_state.show_ref:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Bobot Kriteria (AHP)")
            st.markdown(f"**CR = {cr:.4f} (Konsisten ✅)**")
            bw = pd.DataFrame({"Kriteria": criteria_short, "Bobot": weights.round(4)})
            st.dataframe(bw.set_index("Kriteria"), use_container_width=True)
        with col_b:
            st.markdown("### Tipe Kriteria")
            ct_df = pd.DataFrame(
                {"Kriteria": criteria_short, "Tipe": [ct.upper() for ct in criteria_type]}
            )
            st.dataframe(ct_df.set_index("Kriteria"), use_container_width=True)


with tab3:
    st.subheader("ℹ️ Metodologi AHP + TOPSIS")

    st.markdown(
        """
    **AHP (Analytic Hierarchy Process)** digunakan untuk menentukan bobot
    kepentingan dari 5 kriteria berdasarkan penilaian pakar.
    Bobot dihitung dari matriks perbandingan berpasangan skala Saaty lalu
    diuji konsistensinya (CR < 0.10).

    **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**
    meranking alternatif berdasarkan jarak ke solusi ideal positif (A⁺)
    dan solusi ideal negatif (A⁻). Alternatif terbaik adalah yang terdekat
    ke A⁺ dan terjauh dari A⁻.

    **5 Kriteria:**
    | # | Kriteria | Tipe |
    |---|----------|------|
    | C1 | Efektivitas (%) | Benefit |
    | C2 | Biaya (Rp rb/bln) | Cost |
    | C3 | Kemudahan (1-10) | Benefit |
    | C4 | Efek Samping (1-10) | Cost |
    | C5 | Kecepatan (bln) | Cost |

    **Bobot AHP (CR = {cr:.4f}):**
    | C1 | C2 | C3 | C4 | C5 |
    |----|----|----|----|----|
    | {weights[0]:.4f} | {weights[1]:.4f} | {weights[2]:.4f} | {weights[3]:.4f} | {weights[4]:.4f} |
    """.format(
            cr=cr,
            weights=weights,
        )
    )

    st.markdown("### 📚 Referensi")
    st.markdown(
        """
    1. **ADA**, *Standards of Care in Diabetes—2023*, Diabetes Care, 2023. doi:10.2337/dc23-Srev
    2. **Hwang & Yoon**, *Multiple Attribute Decision Making*, Springer, 1981.
    3. **Jaberidoost et al.**, *JMIR*, 2024. https://medinform.jmir.org/2024/1/e47701
    4. **Wu et al.**, *J. Med. Syst.*, 2018. doi:10.1007/s10916-017-0881-4
    5. **Davies et al.**, *Diabetes Care*, 2022. doi:10.2337/dci22-0034
    """
    )

# ── Popup Referensi Jurnal ──
if st.session_state.show_jurnal:
    with st.sidebar.expander("📚 Referensi Jurnal", expanded=True):
        st.markdown(
            """
        | Kriteria | Referensi |
        |---|---|
        | C1 Efektivitas | ADA, *Standards of Care in Diabetes—2023* |
        | C2 Biaya | Jaberidoost et al., JMIR, 2024 |
        | C3 Kemudahan | Davies et al., Diabetes Care, 2022 |
        | C4 Efek Samping | Wu et al., J.Med.Syst., 2018 |
        | C5 Kecepatan | ADA Clinical Guidelines, 2023 |
        """
        )

st.sidebar.caption("Dibangun dengan Streamlit · AHP+TOPSIS · Python")
