import io
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Gerador de Came -> DXF",
    page_icon="⚙️",
    layout="wide",
)


def calcular_deslocamento_trecho(u, lei):
    """u varia de 0.0 a 1.0 dentro do trecho."""
    u = np.clip(u, 0.0, 1.0)
    if lei == "cicloidal":
        return u - (1.0 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * u)
    elif lei == "polinomial_345":
        return 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
    else:  # mhs
        return 0.5 * (1.0 - np.cos(np.pi * u))


def gerar_perfil_came(raio_base, trechos, lei_selecionada, n_pontos=720):
    """
    trechos: lista de tuplas (theta_ini_deg, theta_fim_deg, tipo, H_mm)
    tipo: 'subida', 'descida' ou 'parada'
    """
    theta_deg = np.linspace(0.0, 360.0, n_pontos, endpoint=True)
    s = np.zeros_like(theta_deg)
    s_acumulado = 0.0
    s_final_esperado = 0.0

    for idx, (t_ini, t_fim, tipo, H) in enumerate(trechos):
        if t_fim <= t_ini:
            continue

        eh_ultimo = idx == len(trechos) - 1
        mask = (theta_deg >= t_ini) & (theta_deg <= t_fim if eh_ultimo else theta_deg < t_fim)
        if not np.any(mask):
            continue

        u = (theta_deg[mask] - t_ini) / (t_fim - t_ini)

        if tipo == "subida":
            s[mask] = s_acumulado + H * calcular_deslocamento_trecho(u, lei_selecionada)
            s_acumulado += H
            s_final_esperado += H
        elif tipo == "descida":
            s[mask] = s_acumulado - H * calcular_deslocamento_trecho(u, lei_selecionada)
            s_acumulado -= H
            s_final_esperado -= H
        else:  # parada
            s[mask] = np.full(np.sum(mask), s_acumulado)

    raio_mm = raio_base + s
    return theta_deg, s, raio_mm, s_final_esperado


def gerar_dxf(theta_deg, raio_mm, raio_base, raio_furo=0.0):
    theta_rad = np.deg2rad(theta_deg)
    x = raio_mm * np.cos(theta_rad)
    y = raio_mm * np.sin(theta_rad)

    pontos = [(float(px), float(py)) for px, py in zip(x, y)]
    if pontos and pontos[0] != pontos[-1]:
        pontos.append(pontos[0])

    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = 4  # Milímetros
    msp = doc.modelspace()

    msp.add_lwpolyline(pontos, close=True, dxfattribs={"layer": "PERFIL_CAME"})
    msp.add_circle((0.0, 0.0), radius=float(raio_base), dxfattribs={"layer": "RAIO_BASE"})

    if raio_furo > 0:
        msp.add_circle((0.0, 0.0), radius=float(raio_furo), dxfattribs={"layer": "FURO_EIXO"})

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode("utf-8")


st.title("⚙️ Perfil Geométrico do Came e Exportação DXF")

st.sidebar.header("Parâmetros Geométricos")
raio_base = st.sidebar.number_input("Raio Base Rb [mm]", min_value=5.0, max_value=300.0, value=27.0, step=1.0)
raio_furo = st.sidebar.number_input("Raio do Furo do Eixo [mm] (opcional)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
lei_selecionada = st.sidebar.selectbox(
    "Lei de Movimento",
    options=["polinomial_345", "cicloidal", "mhs"],
    format_func=lambda x: {
        "polinomial_345": "Polinomial 3-4-5",
        "cicloidal": "Cicloidal",
        "mhs": "Movimento Harmônico Simples (MHS)",
    }[x],
)
n_pontos = st.sidebar.slider("Resolução angular (pontos em 360°)", min_value=180, max_value=2000, value=720, step=60)

st.subheader("Quantidade e Configuração dos Trechos (0° a 360°)")
num_trechos = st.number_input(
    "Quantidade de trechos",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
)

encadear_angulos = st.checkbox(
    "Encadear automaticamente o ângulo inicial ao ângulo final do trecho anterior",
    value=True,
)

# Valores padrão amigáveis para os primeiros trechos
padroes = [
    (0.0, 120.0, "subida", 30.0),
    (120.0, 210.0, "parada", 0.0),
    (210.0, 360.0, "descida", 30.0),
]

trechos = []
angulo_sugerido_ini = 0.0

for i in range(int(num_trechos)):
    padrao_ini, padrao_fim, padrao_tipo, padrao_H = (
        padroes[i] if i < len(padroes) else (min(angulo_sugerido_ini, 350.0), 360.0, "parada", 0.0)
    )
    valor_ini = angulo_sugerido_ini if encadear_angulos else padrao_ini
    valor_fim = 360.0 if (encadear_angulos and i == int(num_trechos) - 1) else max(padrao_fim, min(valor_ini + 10.0, 360.0))

    with st.expander(f"Trecho {i + 1}", expanded=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.4, 1.2])

        with c1:
            t_ini = st.number_input(
                "Ângulo inicial [°]",
                min_value=0.0,
                max_value=360.0,
                value=float(np.clip(valor_ini, 0.0, 360.0)),
                step=5.0,
                key=f"t_ini_{i}",
                disabled=encadear_angulos and i > 0,
            )

        with c2:
            t_fim = st.number_input(
                "Ângulo final [°]",
                min_value=0.0,
                max_value=360.0,
                value=float(np.clip(valor_fim, 0.0, 360.0)),
                step=5.0,
                key=f"t_fim_{i}",
            )

        with c3:
            tipo = st.selectbox(
                "Tipo de movimento",
                options=["subida", "descida", "parada"],
                index=["subida", "descida", "parada"].index(padrao_tipo),
                format_func=lambda t: {
                    "subida": "Subida",
                    "descida": "Descida",
                    "parada": "Parada (Dwell)",
                }[t],
                key=f"tipo_{i}",
            )

        with c4:
            if tipo == "parada":
                st.text_input("Deslocamento H [mm]", value="0.0", disabled=True, key=f"H_parada_{i}")
                H = 0.0
            else:
                H = st.number_input(
                    "Deslocamento H [mm]",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(padrao_H if padrao_H > 0 else 20.0),
                    step=1.0,
                    key=f"H_{i}",
                )

        if t_fim <= t_ini:
            st.error(f"No Trecho {i + 1}, o ângulo final ({t_fim:.1f}°) deve ser maior que o ângulo inicial ({t_ini:.1f}°).")

        trechos.append((float(t_ini), float(t_fim), tipo, float(H)))
        angulo_sugerido_ini = float(t_fim)

if trechos:
    primeiro_ini = trechos[0][0]
    ultimo_fim = trechos[-1][1]
    if not np.isclose(primeiro_ini, 0.0, atol=1e-6) or not np.isclose(ultimo_fim, 360.0, atol=1e-6):
        st.info(
            f"Intervalo coberto pelos trechos: **{primeiro_ini:.1f}°** até **{ultimo_fim:.1f}°**. "
            "Para uma volta completa do came, recomenda-se iniciar em **0.0°** e terminar em **360.0°**."
        )

theta_deg, s, raio_mm, s_final_esperado = gerar_perfil_came(
    raio_base, trechos, lei_selecionada, n_pontos=n_pontos
)

if not np.isclose(s_final_esperado, 0.0, atol=1e-6):
    st.warning(
        f"A soma algébrica dos deslocamentos nos trechos resulta em **{s_final_esperado:+.2f} mm** ao final do ciclo "
        "(subidas somam +H e descidas subtraem -H). Para fechar em s = 0 mm, equilibre subida e descida."
    )

col_graf, col_info = st.columns([2, 1])

with col_graf:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    theta_rad = np.deg2rad(theta_deg)

    ax.plot(theta_rad, raio_mm, color="black", linewidth=2.2, label="Perfil do Came")
    ax.plot(theta_rad, np.full_like(theta_rad, raio_base), "--", color="gray", label="Raio base (Rb)")
    ax.set_title("Perfil Geométrico do Came", pad=18, fontweight="bold")
    ax.legend(loc="upper right")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col_info:
    st.metric("Raio Mínimo", f"{np.min(raio_mm):.2f} mm")
    st.metric("Raio Máximo", f"{np.max(raio_mm):.2f} mm")
    st.metric("Curso Máximo (s)", f"{np.max(s):.2f} mm")

    dxf_bytes = gerar_dxf(theta_deg, raio_mm, raio_base=raio_base, raio_furo=raio_furo)
    st.download_button(
        label="⬇️ Baixar Perfil (.DXF)",
        data=dxf_bytes,
        file_name="perfil_came.dxf",
        mime="application/dxf",
        width="stretch",
    )
