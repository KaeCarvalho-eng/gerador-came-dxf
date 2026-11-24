import io
import re

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Gerador de Came com Rolete -> DXF",
    page_icon="⚙️",
    layout="wide",
)


# =====================================================================
# Cinemática e geometria
# =====================================================================
def calcular_deslocamento_trecho(u, lei):
    """u varia de 0.0 a 1.0 dentro do trecho."""
    u = np.clip(u, 0.0, 1.0)
    if lei == "cicloidal":
        return u - (1.0 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * u)
    elif lei == "polinomial_345":
        return 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
    else:  # mhs
        return 0.5 * (1.0 - np.cos(np.pi * u))


def gerar_perfil_came(trechos, lei_selecionada, n_pontos=720):
    """
    trechos: lista de tuplas (theta_ini_deg, theta_fim_deg, tipo, H_mm)
    tipo: 'subida', 'descida' ou 'parada'
    Retorna theta [graus], s(theta) [mm] e o deslocamento final esperado.
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

    return theta_deg, s, s_final_esperado


def curva_primitiva(theta_deg, s, raio_primitivo, exc):
    """
    Trajetória do centro do rolete no referencial da came.
    P(theta) = R(theta) · (a, e), com a = sqrt(Rp² - e²) + s.
    Para e = 0 reduz-se a r = Rp + s em coordenadas polares.
    """
    th = np.deg2rad(theta_deg)
    y0 = np.sqrt(raio_primitivo**2 - exc**2)
    a = y0 + s
    x = a * np.cos(th) - exc * np.sin(th)
    y = a * np.sin(th) + exc * np.cos(th)
    return x, y, a, y0


def angulo_pressao(theta_deg, s, raio_primitivo, exc):
    """tan(alpha) = (ds/dtheta - e) / (sqrt(Rp² - e²) + s), com ds/dtheta em mm/rad."""
    th = np.deg2rad(theta_deg)
    h = th[1] - th[0]
    ds = np.gradient(s, h, edge_order=2)
    y0 = np.sqrt(raio_primitivo**2 - exc**2)
    alpha = np.degrees(np.arctan((ds - exc) / (y0 + s)))
    return ds, alpha


def contorno_came(x, y, raio_rolete, theta_deg):
    """
    Contorno real da came = curva primitiva deslocada para dentro de Rr
    (offset na direção normal). Retorna também a curvatura da primitiva
    (positiva = convexa), usada para detectar rebaixamento (undercut).
    """
    h = np.deg2rad(theta_deg[1] - theta_deg[0])
    dx = np.gradient(x, h, edge_order=2)
    dy = np.gradient(y, h, edge_order=2)
    ddx = np.gradient(dx, h, edge_order=2)
    ddy = np.gradient(dy, h, edge_order=2)

    # Orientação da curva (área com sinal): +1 anti-horário, -1 horário
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    orient = 1.0 if area >= 0 else -1.0

    norma = np.hypot(dx, dy)
    norma = np.where(norma < 1e-12, 1e-12, norma)
    tx, ty = dx / norma, dy / norma

    # Normal apontando para o interior da curva
    nx, ny = -orient * ty, orient * tx

    cx = x + raio_rolete * nx
    cy = y + raio_rolete * ny
    curvatura = orient * (dx * ddy - dy * ddx) / norma**3
    return cx, cy, curvatura


def gerar_dxf(cx, cy, px, py, raio_primitivo, raio_rolete, exc, raio_furo, extensao_eixo):
    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = 4  # Milímetros

    camadas = {
        "PERFIL_CAME": 7,
        "CURVA_PRIMITIVA": 5,
        "CIRC_PRIMITIVO": 8,
        "CIRC_BASE": 8,
        "FURO_EIXO": 3,
        "ROLETE": 1,
        "EIXO_SEGUIDOR": 1,
    }
    for nome, cor in camadas.items():
        doc.layers.add(name=nome, color=cor)

    msp = doc.modelspace()

    # Descarta o ponto de 360° (coincide com o de 0°); close=True fecha a curva
    contorno = [(float(a), float(b)) for a, b in zip(cx[:-1], cy[:-1])]
    primitiva = [(float(a), float(b)) for a, b in zip(px[:-1], py[:-1])]

    msp.add_lwpolyline(contorno, close=True, dxfattribs={"layer": "PERFIL_CAME"})
    msp.add_lwpolyline(primitiva, close=True, dxfattribs={"layer": "CURVA_PRIMITIVA"})
    msp.add_circle((0.0, 0.0), radius=float(raio_primitivo), dxfattribs={"layer": "CIRC_PRIMITIVO"})

    raio_circ_base = raio_primitivo - raio_rolete
    if raio_circ_base > 0:
        msp.add_circle((0.0, 0.0), radius=float(raio_circ_base), dxfattribs={"layer": "CIRC_BASE"})

    if raio_furo > 0:
        msp.add_circle((0.0, 0.0), radius=float(raio_furo), dxfattribs={"layer": "FURO_EIXO"})

    # Rolete e eixo do seguidor na posição theta = 0°
    msp.add_circle((float(px[0]), float(py[0])), radius=float(raio_rolete), dxfattribs={"layer": "ROLETE"})
    msp.add_line((0.0, float(exc)), (float(extensao_eixo), float(exc)), dxfattribs={"layer": "EIXO_SEGUIDOR"})

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode("utf-8")


# =====================================================================
# Interface
# =====================================================================
st.title("⚙️ Came com Seguidor de Rolete: Perfil, Ângulo de Pressão e DXF")

st.sidebar.header("Parâmetros Geométricos")
raio_base = st.sidebar.number_input(
    "Raio do círculo primitivo Rp [mm]",
    min_value=5.0,
    max_value=300.0,
    value=27.0,
    step=1.0,
    help="Distância do centro da came até o centro do rolete na posição mais baixa (s = 0).",
)
raio_rolete = st.sidebar.number_input("Raio do rolete Rr [mm]", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
exc = st.sidebar.number_input(
    "Excentricidade e [mm]",
    min_value=-150.0,
    max_value=150.0,
    value=0.0,
    step=1.0,
    help=(
        "Deslocamento lateral do eixo do seguidor em relação ao centro da came. "
        "Com a came girando no sentido horário (convenção deste app), e positiva "
        "reduz o ângulo de pressão na subida; negativa aumenta."
    ),
)
raio_furo = st.sidebar.number_input("Raio do Furo do Eixo [mm] (opcional)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
alpha_adm = st.sidebar.number_input("Ângulo de pressão admissível [°]", min_value=5.0, max_value=60.0, value=30.0, step=1.0)
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

if abs(exc) >= raio_base:
    st.error("A excentricidade deve ser menor que o raio do círculo primitivo (|e| < Rp).")
    st.stop()

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

N = int(num_trechos)

# --- Remove o estado dos trechos que não existem mais ---
_padrao_chave = re.compile(r"^(t_ini|t_fim|tipo|H|H_parada)_(\d+)$")
for k in list(st.session_state.keys()):
    m = _padrao_chave.match(k)
    if m and int(m.group(2)) >= N:
        del st.session_state[k]

# --- Pré-passo: inicializa o estado dos trechos novos ANTES de criar os widgets ---
for i in range(N):
    k_ini, k_fim = f"t_ini_{i}", f"t_fim_{i}"
    if k_ini in st.session_state and k_fim in st.session_state:
        continue

    if i < len(padroes):
        st.session_state[k_ini] = padroes[i][0]
        st.session_state[k_fim] = padroes[i][1]
    else:
        # Novo trecho: divide o trecho anterior ao meio para abrir espaço
        prev_ini = st.session_state[f"t_ini_{i - 1}"]
        prev_fim = st.session_state[f"t_fim_{i - 1}"]
        meio = round(((prev_ini + prev_fim) / 2.0) / 5.0) * 5.0
        meio = float(min(max(meio, prev_ini + 5.0), prev_fim - 5.0)) if prev_fim - prev_ini > 10.0 else float(prev_ini)
        st.session_state[f"t_fim_{i - 1}"] = meio
        st.session_state[k_ini] = meio
        st.session_state[k_fim] = float(prev_fim)

trechos = []
angulo_anterior_fim = 0.0

for i in range(N):
    k_ini, k_fim = f"t_ini_{i}", f"t_fim_{i}"
    _, _, padrao_tipo, padrao_H = padroes[i] if i < len(padroes) else (0.0, 0.0, "parada", 0.0)

    eh_ultimo = i == N - 1

    # Força o encadeamento (precisa ser feito antes de instanciar o widget)
    if encadear_angulos:
        st.session_state[k_ini] = 0.0 if i == 0 else float(angulo_anterior_fim)
        if eh_ultimo:
            st.session_state[k_fim] = 360.0

    with st.expander(f"Trecho {i + 1}", expanded=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.4, 1.2])

        with c1:
            t_ini = st.number_input(
                "Ângulo inicial [°]",
                min_value=0.0,
                max_value=360.0,
                step=5.0,
                key=k_ini,
                disabled=encadear_angulos,
            )

        with c2:
            t_fim = st.number_input(
                "Ângulo final [°]",
                min_value=0.0,
                max_value=360.0,
                step=5.0,
                key=k_fim,
                disabled=encadear_angulos and eh_ultimo,
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
        angulo_anterior_fim = float(t_fim)

if trechos:
    primeiro_ini = trechos[0][0]
    ultimo_fim = trechos[-1][1]
    if not np.isclose(primeiro_ini, 0.0, atol=1e-6) or not np.isclose(ultimo_fim, 360.0, atol=1e-6):
        st.info(
            f"Intervalo coberto pelos trechos: **{primeiro_ini:.1f}°** até **{ultimo_fim:.1f}°**. "
            "Para uma volta completa do came, recomenda-se iniciar em **0.0°** e terminar em **360.0°**."
        )

# =====================================================================
# Cálculos
# =====================================================================
theta_deg, s, s_final_esperado = gerar_perfil_came(trechos, lei_selecionada, n_pontos=n_pontos)

if not np.isclose(s_final_esperado, 0.0, atol=1e-6):
    st.warning(
        f"A soma algébrica dos deslocamentos nos trechos resulta em **{s_final_esperado:+.2f} mm** ao final do ciclo "
        "(subidas somam +H e descidas subtraem -H). Para fechar em s = 0 mm, equilibre subida e descida."
    )

px, py, a_arr, y0 = curva_primitiva(theta_deg, s, raio_base, exc)
ds, alpha = angulo_pressao(theta_deg, s, raio_base, exc)
cx, cy, curvatura = contorno_came(px, py, raio_rolete, theta_deg)

# Ignora as pontas (derivadas numéricas unilaterais) nas métricas de curvatura
k_max = float(np.max(curvatura[2:-2]))
rho_min = 1.0 / k_max if k_max > 1e-9 else np.inf
undercut = rho_min <= raio_rolete

raio_circ_base = raio_base - raio_rolete
alpha_max = float(np.max(np.abs(alpha)))
r_contorno = np.hypot(cx, cy)

# Avisos
if raio_circ_base <= 0:
    st.error("O raio do rolete é maior ou igual ao raio primitivo: não há material de came. Aumente Rp ou reduza Rr.")
elif raio_furo >= float(np.min(r_contorno)):
    st.warning(
        f"O raio do furo ({raio_furo:.1f} mm) é maior ou igual ao menor raio do contorno ({np.min(r_contorno):.1f} mm)."
    )

if undercut:
    st.error(
        f"**Rebaixamento (undercut):** o menor raio de curvatura da curva primitiva é "
        f"**{rho_min:.2f} mm**, menor ou igual ao raio do rolete ({raio_rolete:.2f} mm). "
        "O contorno se auto-intersecta e a came não gera o movimento desejado. "
        "Aumente Rp, reduza Rr, aumente o ângulo do trecho ou reduza H."
    )

if alpha_max > alpha_adm:
    st.warning(
        f"O ângulo de pressão máximo (**{alpha_max:.1f}°**) excede o admissível ({alpha_adm:.0f}°). "
        "Aumente Rp, aumente o ângulo do trecho, use outra lei de movimento ou ajuste a excentricidade."
    )

# =====================================================================
# Visualização
# =====================================================================
theta_sel = st.slider("Posição da came θ [°]", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

s_sel = float(np.interp(theta_sel, theta_deg, s))
alpha_sel = float(np.interp(theta_sel, theta_deg, alpha))
th_sel = np.deg2rad(theta_sel)
a_sel = y0 + s_sel


def girar(vx, vy):
    """Gira um vetor do referencial do seguidor para o da came (ângulo theta_sel)."""
    return vx * np.cos(th_sel) - vy * np.sin(th_sel), vx * np.sin(th_sel) + vy * np.cos(th_sel)


xc, yc = girar(a_sel, exc)

col_graf, col_info = st.columns([2, 1])

with col_graf:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    t_circ = np.linspace(0.0, 2.0 * np.pi, 361)

    ax.plot(px, py, "--", color="tab:blue", linewidth=1.2, label="Curva primitiva")
    ax.plot(cx, cy, color="black", linewidth=2.2, label="Contorno da came")
    ax.plot(raio_base * np.cos(t_circ), raio_base * np.sin(t_circ), ":", color="gray", linewidth=1.0, label="Círculo primitivo")
    if raio_furo > 0:
        ax.plot(raio_furo * np.cos(t_circ), raio_furo * np.sin(t_circ), color="tab:green", linewidth=1.2, label="Furo do eixo")

    # Rolete e haste do seguidor na posição selecionada
    ax.add_patch(plt.Circle((xc, yc), raio_rolete, fill=False, color="tab:red", linewidth=2.0))
    ax.plot([xc], [yc], "o", color="tab:red", markersize=4)
    L = 0.5 * raio_base
    hx, hy = girar(np.array([a_sel - 0.0, a_sel + L + raio_rolete]), np.array([exc, exc]))
    ax.plot(hx, hy, "-", color="tab:red", linewidth=1.5, label="Eixo do seguidor")
    ax.plot([0.0], [0.0], "+", color="black", markersize=10)

    lim = float(np.max(np.hypot(px, py))) + raio_rolete + 0.5 * raio_base
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"θ = {theta_sel:.0f}°   s = {s_sel:.2f} mm   α = {alpha_sel:.1f}°", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col_info:
    st.metric("Curso Máximo (s)", f"{np.max(s):.2f} mm")
    st.metric("Raio Mínimo do Contorno", f"{np.min(r_contorno):.2f} mm")
    st.metric("Raio Máximo do Contorno", f"{np.max(r_contorno):.2f} mm")
    st.metric("Ângulo de Pressão Máximo", f"{alpha_max:.1f} °")
    st.metric(
        "Menor Raio de Curvatura (primitiva)",
        "convexo livre" if np.isinf(rho_min) else f"{rho_min:.2f} mm",
    )

    extensao = float(np.max(np.hypot(px, py))) + raio_rolete + 10.0
    dxf_bytes = gerar_dxf(cx, cy, px, py, raio_base, raio_rolete, exc, raio_furo, extensao)
    st.download_button(
        label="⬇️ Baixar Perfil (.DXF)",
        data=dxf_bytes,
        file_name="perfil_came_rolete.dxf",
        mime="application/dxf",
        width="stretch",
    )

# Gráficos de deslocamento e ângulo de pressão
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))

ax1.plot(theta_deg, s, color="black", linewidth=2.0)
ax1.axvline(theta_sel, color="tab:red", linewidth=1.0)
ax1.plot([theta_sel], [s_sel], "o", color="tab:red")
ax1.set_xlim(0, 360)
ax1.set_xlabel("θ [°]")
ax1.set_ylabel("s [mm]")
ax1.set_title("Deslocamento do seguidor")
ax1.grid(True, alpha=0.3)

ax2.plot(theta_deg, alpha, color="black", linewidth=2.0)
ax2.axhline(alpha_adm, color="tab:orange", linestyle="--", linewidth=1.0)
ax2.axhline(-alpha_adm, color="tab:orange", linestyle="--", linewidth=1.0, label=f"±{alpha_adm:.0f}° (admissível)")
ax2.axvline(theta_sel, color="tab:red", linewidth=1.0)
ax2.plot([theta_sel], [alpha_sel], "o", color="tab:red")
ax2.set_xlim(0, 360)
ax2.set_xlabel("θ [°]")
ax2.set_ylabel("α [°]")
ax2.set_title("Ângulo de pressão")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=8)

fig2.tight_layout()
st.pyplot(fig2, width="stretch")
plt.close(fig2)
