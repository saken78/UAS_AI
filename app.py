import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SPK Diabetes — AHP+TOPSIS", page_icon="", layout="wide"
)


# ── Load Model dari Joblib (Bobot AHP dari notebook) ────────
@st.cache_resource
def load_model():
    return joblib.load("model_spk_diabetes.joblib")


model = load_model()
base_weights = model["weights"]
criteria_type = model["criteria_type"]
cr = 0.0173

# ── Knowledge Base ──────────────────────────────────────────
criteria_short = ["Efektivitas", "Biaya", "Kemudahan", "Efek Samping", "Kecepatan"]
criteria_unit = ["%", "Rp rb/bln", "1-10", "1-10", "bln"]

alt_names = [
    "A1: Diet Rendah Karb + Olahraga",
    "A2: Konsultasi Dokter + HbA1c",
    "A3: Metformin + Monitor Gula Darah",
    "A4: Program Penurunan Berat Badan",
    "A5: Intervensi Gaya Hidup Intensif",
]
alt_short = [
    "Diet+Olahraga",
    "Konsultasi Dokter",
    "Metformin",
    "Penurunan BB",
    "Gaya Hidup Intensif",
]

ref_matrix = np.array(
    [
        [80, 150, 8, 2, 3],  # A1
        [75, 200, 6, 1, 1],  # A2
        [90, 350, 7, 5, 1],  # A3
        [70, 100, 9, 1, 4],  # A4
        [88, 500, 5, 2, 2],  # A5
    ],
    dtype=float,
)


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


# ── Session State ───────────────────────────────────────────
if "show_jurnal" not in st.session_state:
    st.session_state.show_jurnal = False
if "show_ref" not in st.session_state:
    st.session_state.show_ref = False

# ── Sidebar: Input Data Klinis Pasien ───────────────────────
st.sidebar.title("Input Data Klinis Pasien")
st.sidebar.markdown(
    "<small>Masukkan kondisi pasien untuk personalisasi rekomendasi.</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

usia = st.sidebar.number_input("Usia (tahun)", min_value=20, max_value=100, value=55)
bb = st.sidebar.number_input(
    "Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0
)
tb = st.sidebar.number_input(
    "Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=165.0
)
hba1c = st.sidebar.number_input(
    "HbA1c Terakhir (%)", min_value=5.0, max_value=15.0, value=8.5, step=0.1
)
budget = st.sidebar.selectbox(
    "Kisaran Budget per Bulan",
    ["< Rp 100.000", "Rp 100.000 - 300.000", "> Rp 300.000"],
)

# Hitung BMI
bmi = bb / ((tb / 100) ** 2)
if bmi >= 30:
    status_bmi = "Obesitas"
elif bmi >= 25:
    status_bmi = "Overweight"
elif bmi >= 18.5:
    status_bmi = "Normal"
else:
    status_bmi = "Underweight"

# Tombol referensi
if st.sidebar.button("Lihat Referensi Jurnal"):
    st.session_state.show_jurnal = not st.session_state.show_jurnal

# ── Dynamic Weighting (Aturan Klinis) ───────────────────────
adjusted_weights = base_weights.copy()
warnings = []

# Aturan 1: HbA1c sangat tinggi → Efektivitas mutlak
if hba1c > 9.0:
    adjusted_weights[0] += 0.15  # C1 Efektivitas naik
    adjusted_weights[4] -= 0.15  # C5 Kecepatan turun
    warnings.append(
        "**HbA1c Tinggi (>9.0%)**: Efektivitas diprioritaskan — kecepatan bukan fokus."
    )

# Aturan 2: Budget terbatas → Biaya jadi prioritas
if budget == "< Rp 100.000":
    adjusted_weights[1] += 0.20  # C2 Biaya naik
    adjusted_weights[4] -= 0.10  # C5 Kecepatan turun
    warnings.append(
        "**Budget Terbatas (< Rp 100rb)**: Opsi terjangkau diprioritaskan."
    )

# Aturan 3: Obesitas → Program murah & mudah
if bmi >= 30:
    adjusted_weights[1] += 0.10  # C2 Biaya naik
    adjusted_weights[2] += 0.10  # C3 Kemudahan naik
    adjusted_weights[4] -= 0.20  # C5 Kecepatan turun
    warnings.append(
        "**Obesitas (BMI > 30)**: Program terjangkau & mudah diprioritaskan."
    )

# Aturan 4: Lansia → Minim efek samping
if usia > 65:
    adjusted_weights[3] += 0.15  # C4 Efek Samping naik
    adjusted_weights[4] -= 0.15  # C5 Kecepatan turun
    warnings.append("**Lansia (Usia > 65)**: Minim efek samping diprioritaskan.")

# Pastikan tidak ada bobot negatif, lalu normalisasi
adjusted_weights = np.clip(adjusted_weights, 0, None)
adjusted_weights = adjusted_weights / adjusted_weights.sum()

# ── Main Content ────────────────────────────────────────────
st.title("SPK Diabetes Melitus Tipe 2")
st.markdown(
    "Sistem Pendukung Keputusan — Metode **AHP + TOPSIS** dengan *Dynamic Weighting*"
)
st.caption(
    "Rekomendasi penatalaksanaan dipersonalisasi berdasarkan profil klinis pasien."
)

# Tampilkan warning
if warnings:
    for w in warnings:
        st.warning(w)

# ── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["Hasil & Rekomendasi", "Data Alternatif", "Metodologi & Bobot"]
)

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    # ── Jalankan TOPSIS (5 alternatif) ──
    scores = topsis(ref_matrix, adjusted_weights, criteria_type)

    # Ranking
    df_result = pd.DataFrame(
        {
            "Alternatif": alt_names,
            "Skor TOPSIS": np.round(scores, 5),
        }
    )
    df_result["Ranking"] = df_result["Skor TOPSIS"].rank(ascending=False).astype(int)
    df_result = df_result.sort_values("Ranking").reset_index(drop=True)
    df_result["Ranking"] = df_result["Ranking"].apply(
        lambda r: ["", "", "", "", ""][r - 1] + f" Rank {r}"
    )

    with col1:
        st.subheader("Hasil Perankingan")
        st.dataframe(
            df_result.set_index("Alternatif"),
            use_container_width=True,
            column_config={
                "Skor TOPSIS": st.column_config.NumberColumn(format="%.5f"),
            },
        )

        best_idx = np.argmax(scores)
        st.success(
            f"**Rekomendasi terbaik untuk pasien ini:**\n\n {alt_names[best_idx]}"
        )
        st.balloons()

    with col2:
        st.subheader("Visualisasi Skor")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#2E74B5", "#4BACC6", "#A9D18E", "#FFD966", "#F4B183"]

        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        sorted_labels = [alt_short[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]

        bars = ax.barh(
            range(5), sorted_scores, color=sorted_colors, edgecolor="white", height=0.6
        )
        ax.set_yticks(range(5))
        ax.set_yticklabels(sorted_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Skor TOPSIS")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, sc in zip(bars, sorted_scores):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sc:.4f}",
                va="center",
                fontsize=9,
            )
        ax.set_title("Skor TOPSIS — 5 Alternatif Penatalaksanaan", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

with tab2:
    st.subheader("5 Alternatif Penatalaksanaan Diabetes (Data Referensi)")
    df_alt = pd.DataFrame(
        ref_matrix.astype(int),
        index=alt_short,
        columns=criteria_short,
    )
    st.dataframe(df_alt, use_container_width=True)
    st.info(
        "*Data bersifat tetap dari sintesis literatur klinis (ADA 2023). "
        "Yang berubah sesuai pasien adalah **bobot prioritas** kriteria.*"
    )

with tab3:
    st.subheader("Metodologi AHP + TOPSIS & Dynamic Weighting")

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
    """
    )

    st.markdown(f"""

    **5 Kriteria Keputusan:**

    | # | Kriteria | Tipe | Deskripsi |
    |---|----------|------|-----------|
    | C1 | Efektivitas (%) | Benefit | Persentase keberhasilan penatalaksanaan |
    | C2 | Biaya (Rp rb/bln) | Cost | Estimasi biaya per bulan (ribu rupiah) |
    | C3 | Kemudahan (1-10) | Benefit | Skor kemudahan penerapan |
    | C4 | Efek Samping (1-10) | Cost | Skor potensi efek samping |
    | C5 | Kecepatan (bln) | Cost | Estimasi waktu sampai hasil terlihat |

    """)

    st.markdown("### Aturan Dynamic Weighting")
    st.markdown("""
    | Aturan | Kondisi | Efek Bobot | Alasan Klinis |
    |--------|---------|------------|---------------|
    | R1 | HbA1c > 9.0% | C1 +0.15, C5 -0.15 | Gula sangat tinggi → efektivitas mutlak |
    | R2 | Budget < Rp 100.000 | C2 +0.20, C5 -0.10 | Budget terbatas → biaya jadi prioritas |
    | R3 | BMI >= 30 (Obesitas) | C2 +0.10, C3 +0.10, C5 -0.20 | Obesitas → program murah & mudah |
    | R4 | Usia > 65 (Lansia) | C4 +0.15, C5 -0.15 | Lansia → minim efek samping |
    """)

    st.markdown("### Referensi")
    st.markdown(
        """
    1. **ADA**, *Standards of Care in Diabetes—2023*, Diabetes Care, 2023. doi:10.2337/dc23-Srev
    2. **Hwang & Yoon**, *Multiple Attribute Decision Making: Methods and Applications*, Springer, 1981.
    3. **Jaberidoost et al.**, *Evaluation of Machine Learning-Based Models for Predicting Diabetes*, JMIR, 2024. https://medinform.jmir.org/2024/1/e47701
    4. **Wu et al.**, *An Integrated Approach for Clinical Decision Support*, J. Med. Syst., 2018. doi:10.1007/s10916-017-0881-4
    5. **Davies et al.**, *Management of Hyperglycemia in Type 2 Diabetes, 2022*, Diabetes Care, 2022. doi:10.2337/dci22-0034
    """
    )

# ── Popup Referensi Jurnal ──
if st.session_state.show_jurnal:
    with st.sidebar.expander("Referensi Jurnal", expanded=True):
        st.markdown("""
        | Kriteria | Referensi |
        |---|---|
        | C1 Efektivitas | ADA, *Standards of Care in Diabetes—2023* |
        | C2 Biaya | Jaberidoost et al., JMIR, 2024 |
        | C3 Kemudahan | Davies et al., Diabetes Care, 2022 |
        | C4 Efek Samping | Wu et al., J.Med.Syst., 2018 |
        | C5 Kecepatan | ADA Clinical Guidelines, 2023 |
        """)

st.sidebar.caption("Dibangun dengan Streamlit · AHP+TOPSIS · Python")
