"""Analisa amostras.csv da calibração e deriva os limiares do detect.py.

Fecha o ciclo de calibração:
  calibracao_temperatura.py  ->  amostras.csv  ->  ESTE script  ->  detect.py

A assinatura útil é o CONTRASTE do alvo contra o fundo, |ΔT| = |T_alvo − T_amb|,
que funciona nos DOIS regimes (cômodo frio: pessoa quente, ΔT>0; cômodo quente:
pessoa fria, ΔT<0). A envoltória INFERIOR de |ΔT| nas amostras de PESSOA VIVA é
o contraste mínimo que uma pessoa produz em cada nível de resolução — ou seja, o
limiar de "vivo".

Sobre a curva ajustada
----------------------
O detect.py não usa mais um degrau de dois níveis: usa a curva contínua de
termica_comum.limiar_dt_vivo, e é ELA que este script ajusta à envoltória. A
área de saturação (AREA_BEM_RESOLVIDO) NÃO é ajustada aos dados — vem fixa da
geometria do sensor (IFOV) ancorada numa distância operacional. Derivá-la da
mediana das áreas coletadas faria a fronteira andar conforme onde você coletou,
que é artefato de amostragem e não física do alvo.

A temperatura do alvo é a cauda da região mais distante do ambiente (p01 no
regime quente, p99 no frio). Formatos aceitos: novo (t_p99/t_p01 + metadados) e
antigo (t_alvo_c = p99, tratado como regime frio, sem metadados).

Positivos x negativos
---------------------
A curva acima diz o que uma PESSOA produz. Ela sozinha não mede nada sobre falso
positivo: para isso é preciso saber o que um NÃO-humano produz na mesma régua.
As amostras 'negativo_*' entram aqui exatamente para isso, e delas saem os dois
parâmetros da fusão do detect.py (P_TERMICA_MIN/P_TERMICA_MAX) — quanto a
térmica pode empurrar a decisão em cada sentido.

Saídas: 9 gráficos em calibracao/dados/curva_calibracao.png e, no terminal,
os parâmetros prontos para colar no topo do detect.py.
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib

os.environ["QT_QPA_PLATFORM"] = "xcb"  # evita warning do Wayland (igual plotar.py)
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
# termica_comum.py fica na pasta de cima, junto do detect.py: os dois PRECISAM usar o mesmo
# módulo, senão a calibração deixa de valer para o detector.
PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PASTA_PROJETO)
PASTA_DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")

from termica_comum import (
    DISTANCIA_MIN_ENQUADRA_M,
    area_esperada_px,
    contraste_direcionado,
    fundir_confianca,
    limiar_dt_vivo,
    plausibilidade_termica,
)

ARQUIVO_CSV = os.path.join(PASTA_DADOS, "amostras.csv")
PERCENTIL_ENVOLTORIA = 10   # borda inferior da nuvem por faixa de área
MARGEM_SEGURANCA = 0.5      # °C abaixo da envoltória p/ não rejeitar vivo
N_BINS_ENVOLTORIA = 6
MIN_POR_BIN = 3
# Fronteiras de regime (devem casar com o detect.py). Marcadas no gráfico 3.
T_PELE_MIN = 33.0
T_PELE_MAX = 36.0
# Mesma âncora do detect.py: a saturação da curva vem da geometria, não do dado.
DISTANCIA_BEM_RESOLVIDO_M = 4.5
AREA_BEM_RESOLVIDO = area_esperada_px(DISTANCIA_BEM_RESOLVIDO_M)
# Barra de aceite do detect.py, usada aqui só para contar resgates/quedas.
LIMIAR_ACEITE = 0.50
CONF_YOLO_SUFICIENTE = 0.50
# Piso/teto impostos à plausibilidade estimada. Existem porque a estimativa vem
# de uma amostra finita: com 20 negativos e nenhum deles caindo na faixa de alto
# contraste, a conta dá p=1.0 (veto absoluto no sentido oposto), que é confiança
# que o tamanho da amostra não sustenta. Aparar evita que a térmica ganhe poder
# de decisão que os dados não deram.
P_LIMITE_INFERIOR = 0.20
P_LIMITE_SUPERIOR = 0.90


def _tail_alvo(t_p99, t_p01, t_amb):
    """Temperatura do alvo = cauda da região mais distante do ambiente."""
    return t_p99 if abs(t_p99 - t_amb) >= abs(t_p01 - t_amb) else t_p01


def carregar(caminho):
    """Lê o CSV (';'), aceitando formato novo e antigo. Devolve dict de arrays."""
    campos = defaultdict(list)
    n_descartadas_amb = 0
    with open(caminho, "r", newline="") as f:
        for lin in csv.DictReader(f, delimiter=";"):
            try:
                ta = float(lin["t_ambiente_c"])
                if lin.get("t_p99_c"):                       # formato novo
                    tp = _tail_alvo(float(lin["t_p99_c"]), float(lin["t_p01_c"]), ta)
                else:                                        # formato antigo
                    tp = float(lin["t_alvo_c"])  # era p99 -> assume regime frio
                # Amostra com ambiente não estimável (alvo enchendo o frame) tem
                # ΔT artificialmente colapsado: contamina a envoltória.
                if lin.get("amb_confiavel", "1") == "0":
                    n_descartadas_amb += 1
                    continue
                campos["dist"].append(float(lin["distancia_m"]))
                campos["area"].append(float(lin["area_px_termico"]))
                campos["t_alvo"].append(tp)
                campos["t_amb"].append(ta)
                campos["dt"].append(tp - ta)  # contraste COM sinal
                campos["classe"].append(lin.get("classe") or "viva")
                campos["pose"].append(lin.get("pose") or "?")
                campos["roupa"].append(lin.get("roupa") or "?")
                campos["local"].append(lin.get("local") or "?")
                campos["t_marco"].append(float(lin["t_desde_marco_s"])
                                         if lin.get("t_desde_marco_s") else np.nan)
                campos["conf"].append(float(lin["conf"]) if lin.get("conf")
                                      else np.nan)
            except (KeyError, ValueError):
                continue
    dados = {k: np.array(v) for k, v in campos.items()}
    return dados, n_descartadas_amb


def envoltoria_inferior(area, abs_dt, n_bins=N_BINS_ENVOLTORIA,
                        pct=PERCENTIL_ENVOLTORIA):
    """Percentil baixo de |ΔT| por faixa de área -> pontos da envoltória.

    Bins em ESCALA LOG: a área varia por mais de uma ordem de grandeza entre 1 m
    e 12 m, e bins lineares jogariam quase toda a amostra distante no primeiro
    bin — justamente a faixa que define o limiar de alcance máximo.
    """
    if len(area) < 2 or area.max() <= 0:
        return np.array([]), np.array([])
    positivas = area[area > 0]
    bordas = np.geomspace(positivas.min(), positivas.max(), n_bins + 1)
    centros, dt_baixo = [], []
    for i in range(n_bins):
        m = (area >= bordas[i]) & (area <= bordas[i + 1])
        if m.sum() >= MIN_POR_BIN:
            centros.append(np.sqrt(bordas[i] * bordas[i + 1]))  # centro geométrico
            dt_baixo.append(np.percentile(abs_dt[m], pct))
    return np.array(centros), np.array(dt_baixo)


def modelo_limiar(area, dt_perto, dt_longe, expoente):
    """Mesma forma de termica_comum.limiar_dt_vivo, vetorizada p/ o ajuste."""
    razao = np.clip(np.asarray(area, dtype=float) / AREA_BEM_RESOLVIDO, 0, 1) ** expoente
    return dt_longe + (dt_perto - dt_longe) * razao


def ajustar_curva(centros, dt_env):
    """Ajusta (dt_perto, dt_longe, expoente) à envoltória. None se faltar dado."""
    if len(centros) < 3:
        return None
    try:
        popt, _ = curve_fit(
            modelo_limiar, centros, dt_env,
            p0=[max(dt_env), min(dt_env), 0.5],
            bounds=([0.0, 0.0, 0.1], [30.0, 30.0, 3.0]), maxfev=20000)
        return tuple(float(p) for p in popt)
    except (RuntimeError, ValueError):
        return None


def regime_de(t_amb):
    """Regime térmico de cada amostra (mesmas fronteiras do detect.py)."""
    return np.where(t_amb < T_PELE_MIN, "frio",
                    np.where(t_amb > T_PELE_MAX, "quente", "crossover"))


def razao_contraste(dt, t_amb, area, ajuste):
    """Contraste do alvo em unidades do limiar do PRÓPRIO tamanho.

    Normalizar pelo limiar é o que torna comparáveis uma pessoa a 2 m e outra a
    10 m: em °C brutos a distante sempre parece mais fraca (spot-size), e a
    separação positivo/negativo ficaria confundida com a distância da coleta.
    Em unidades de limiar, 1.0 significa "produziu o contraste que se espera de
    uma pessoa deste tamanho aparente", em qualquer distância.
    """
    dt_perto, dt_longe, expoente = ajuste
    razoes = np.full(len(dt), np.nan)
    regimes = regime_de(t_amb)
    for i, (d, r, a) in enumerate(zip(dt, regimes, area)):
        if r == "crossover":
            continue        # sem sentido esperado de contraste: não entra
        limiar = limiar_dt_vivo(a, dt_perto, dt_longe, AREA_BEM_RESOLVIDO, expoente)
        if limiar <= 0:
            continue
        razoes[i] = contraste_direcionado(d, r) / limiar
    return razoes


def estimar_plausibilidade(razao, eh_viva, eh_negativa):
    """(p_min, p_max) empíricos: P(pessoa | assinatura) nos dois extremos.

    p_max é lido na faixa de contraste PLENO (razão >= 1) e p_min na faixa SEM
    contraste no sentido esperado (razão <= 0). Em cada faixa a conta é a
    proporção de amostras que de fato eram pessoa.

    Isso é a posterior sob prior 50/50, que é exatamente o que a fusão em razão
    de chances assume ao multiplicar p/(1−p) pelas chances do YOLO. A leitura só
    é válida se positivos e negativos tiverem sido coletados em quantidade
    parecida — coletar 200 pessoas e 5 manequins infla p_min sozinho. O aviso
    correspondente é impresso quando o desbalanço é grande.

    Devolve (p_min, p_max, diagnóstico) — p_* = None quando falta amostra.
    """
    def proporcao(mascara_faixa):
        nv = int((eh_viva & mascara_faixa).sum())
        nn = int((eh_negativa & mascara_faixa).sum())
        if nv + nn < MIN_POR_BIN:
            return None, nv, nn
        p = nv / (nv + nn)
        return float(np.clip(p, P_LIMITE_INFERIOR, P_LIMITE_SUPERIOR)), nv, nn

    valida = ~np.isnan(razao)
    p_max, nv_alto, nn_alto = proporcao(valida & (razao >= 1.0))
    p_min, nv_baixo, nn_baixo = proporcao(valida & (razao <= 0.0))
    diag = dict(nv_alto=nv_alto, nn_alto=nn_alto,
                nv_baixo=nv_baixo, nn_baixo=nn_baixo)
    return p_min, p_max, diag


def resumo_por(rotulo, chaves, abs_dt):
    """Imprime |ΔT| mediano por grupo (vestuário, pose, local)."""
    grupos = sorted(set(chaves))
    if len(grupos) <= 1 and grupos[:1] in ([], ["?"]):
        return
    print(f"\n|ΔT| mediano por {rotulo}:")
    for g in grupos:
        m = chaves == g
        print(f"  {g:<18} n={m.sum():<4} mediana={np.median(abs_dt[m]):.2f}°C  "
              f"min={abs_dt[m].min():.2f}  max={abs_dt[m].max():.2f}")


def main():
    if not os.path.exists(ARQUIVO_CSV):
        print(f"Erro: '{ARQUIVO_CSV}' não encontrado. Rode a calibração antes.")
        return

    dados, n_desc = carregar(ARQUIVO_CSV)
    if len(dados.get("dt", [])) == 0:
        print("Nenhuma amostra válida no CSV.")
        return
    if n_desc:
        print(f"[i] {n_desc} amostra(s) descartada(s): ambiente não estimável "
              f"(alvo ocupava o frame).")

    abs_dt = np.abs(dados["dt"])
    t_amb, area, dist = dados["t_amb"], dados["area"], dados["dist"]
    classe = dados["classe"]

    # --- Panorama da coleta ------------------------------------------------
    n_frio = int((t_amb < T_PELE_MIN).sum())
    n_cross = int(((t_amb >= T_PELE_MIN) & (t_amb <= T_PELE_MAX)).sum())
    n_quente = int((t_amb > T_PELE_MAX).sum())
    print(f"\n{len(abs_dt)} amostras | distâncias: {sorted(set(np.round(dist, 1)))} m")
    print(f"Regimes -> frio:{n_frio}  crossover:{n_cross}  quente:{n_quente}")
    print("Classes -> " + "  ".join(f"{c}:{int((classe == c).sum())}"
                                    for c in sorted(set(classe))))
    if n_quente == 0:
        print("  [!] Sem cômodo QUENTE (t_amb>36°C): o regime quente do detect.py "
              "fica sem validação empírica.")

    # --- Envoltória e ajuste: SÓ pessoa viva -------------------------------
    viva = classe == "viva"
    if viva.sum() < MIN_POR_BIN:
        print("  [!] Amostras de 'viva' insuficientes para ajustar a curva.")
        return
    area_v, abs_dt_v, dist_v = area[viva], abs_dt[viva], dist[viva]

    cen, dt_env = envoltoria_inferior(area_v, abs_dt_v)
    ajuste = ajustar_curva(cen, dt_env)

    print("\n--- Parâmetros para o topo do detect.py ---")
    print(f"DISTANCIA_BEM_RESOLVIDO_M = {DISTANCIA_BEM_RESOLVIDO_M}   "
          f"# -> AREA_BEM_RESOLVIDO = {AREA_BEM_RESOLVIDO:.0f} px (via IFOV)")
    if ajuste is not None:
        dt_perto, dt_longe, expoente = ajuste
        dt_perto_s = max(0.0, dt_perto - MARGEM_SEGURANCA)
        dt_longe_s = max(0.0, dt_longe - MARGEM_SEGURANCA)
        print(f"DT_VIVO_PERTO = {dt_perto_s:.1f}")
        print(f"DT_VIVO_LONGE = {dt_longe_s:.1f}")
        print(f"EXPOENTE_LIMIAR = {expoente:.2f}")
        if dt_perto <= dt_longe:
            print("  [!] Ajuste NÃO monotônico (perto <= longe): o contraste não "
                  "cresce com a resolução do alvo. Sinal de variável não "
                  "controlada (vestuário/pose/NUC) ou de faixa de distância curta "
                  "demais. NÃO cole esses números antes de investigar.")
    else:
        print("  [!] Envoltória com menos de 3 faixas de área — colete mais "
              "distâncias antes de ajustar a curva. Estimativa grosseira abaixo:")
        corte = area_v >= AREA_BEM_RESOLVIDO
        if corte.any():
            print(f"      DT_VIVO_PERTO ~ "
                  f"{max(0.0, np.percentile(abs_dt_v[corte], PERCENTIL_ENVOLTORIA) - MARGEM_SEGURANCA):.1f}")
        if (~corte).any():
            print(f"      DT_VIVO_LONGE ~ "
                  f"{max(0.0, np.percentile(abs_dt_v[~corte], PERCENTIL_ENVOLTORIA) - MARGEM_SEGURANCA):.1f}")

    print(f"\n(T_alvo observada: {dados['t_alvo'].min():.1f}–{dados['t_alvo'].max():.1f}°C | "
          f"ambiente: {t_amb.min():.1f}–{t_amb.max():.1f}°C)")

    # --- Parâmetros da FUSÃO: exigem os negativos --------------------------
    # A curva acima diz o que uma pessoa produz. Isto aqui diz o quanto essa
    # informação vale contra a forma do YOLO — e só é estimável comparando com
    # o que um NÃO-humano produz na mesma régua.
    negativa = np.char.startswith(classe.astype(str), "negativo")
    ajuste_ou_padrao = ajuste if ajuste is not None else (3.0, 1.0, 0.5)
    razao = razao_contraste(dados["dt"], t_amb, area, ajuste_ou_padrao)
    p_min, p_max, diag = estimar_plausibilidade(razao, viva, negativa)

    print("\n--- Parâmetros da fusão (reforço da YOLO) ---")
    if negativa.sum() == 0:
        print("  [!] NENHUMA amostra 'negativo_*'. Sem elas não há como estimar "
              "P_TERMICA_MIN/MAX nem apresentar taxa de falso positivo: a térmica "
              "fica sem medida do próprio poder de derrubar caixa errada.")
        print("      Colete manequim/casaco pendurado, fonte de calor e reflexo.")
    else:
        razao_neg = razao[negativa & ~np.isnan(razao)]
        razao_pos = razao[viva & ~np.isnan(razao)]
        desbalanco = max(viva.sum(), negativa.sum()) / max(1, min(viva.sum(),
                                                                 negativa.sum()))
        if p_max is not None:
            print(f"P_TERMICA_MAX = {p_max:.2f}   "
                  f"# faixa razão>=1: {diag['nv_alto']} vivas x {diag['nn_alto']} negativas")
        else:
            print(f"  [!] Amostras de MAS insuficientes na faixa de contraste pleno "
                  f"({diag['nv_alto']} vivas, {diag['nn_alto']} negativas).")
        if p_min is not None:
            print(f"P_TERMICA_MIN = {p_min:.2f}   "
                  f"# faixa razão<=0: {diag['nv_baixo']} vivas x {diag['nn_baixo']} negativas")
        else:
            print(f"  [!] Amostras insuficientes na faixa sem contraste "
                  f"({diag['nv_baixo']} vivas, {diag['nn_baixo']} negativas).")
        if len(razao_pos) and len(razao_neg):
            print(f"    razão de contraste — viva: mediana {np.median(razao_pos):+.2f} | "
                  f"negativa: mediana {np.median(razao_neg):+.2f}")
            if np.median(razao_pos) <= np.median(razao_neg):
                print("  [!] Negativos com contraste IGUAL ou MAIOR que as pessoas. "
                      "A térmica não separa as duas classes nesta coleta — antes de "
                      "colar qualquer número, verifique se os negativos não são "
                      "todos fonte de calor (que o veto já elimina) em vez de "
                      "objetos passivos com forma de gente.")
        if desbalanco > 3:
            print(f"  [!] Coleta desbalanceada ({int(viva.sum())} vivas x "
                  f"{int(negativa.sum())} negativas). A estimativa supõe prior 50/50: "
                  f"com esse desbalanço ela puxa os dois valores para a classe "
                  f"majoritária. Equilibre antes de usar.")

        # Efeito da fusão, amostra a amostra: o que a térmica teria mudado.
        conf = dados["conf"]
        if p_min is not None and p_max is not None and not np.all(np.isnan(conf)):
            # Mesmas funções que o detect.py usa em tempo real. Reimplementar a
            # plausibilidade aqui faria a análise medir um detector que não é o
            # que roda — é o motivo de termica_comum existir.
            score = np.array([
                fundir_confianca(c, plausibilidade_termica(r, 1.0, p_min, p_max))
                if not (np.isnan(c) or np.isnan(r)) else np.nan
                for c, r in zip(conf, razao)])
            ok = ~np.isnan(score)
            resgates = int((viva & ok & (conf < CONF_YOLO_SUFICIENTE)
                            & (score >= LIMIAR_ACEITE)).sum())
            perdidas = int((viva & ok & (conf >= CONF_YOLO_SUFICIENTE)
                            & (score < LIMIAR_ACEITE)).sum())
            derrubados = int((negativa & ok & (conf >= CONF_YOLO_SUFICIENTE)
                              & (score < LIMIAR_ACEITE)).sum())
            mantidos = int((negativa & ok & (score >= LIMIAR_ACEITE)).sum())
            print("\n  Efeito da fusão sobre esta amostra:")
            print(f"    pessoas RESGATADAS (yolo<{CONF_YOLO_SUFICIENTE:.2f} -> aceita): {resgates}")
            print(f"    negativos DERRUBADOS (yolo forte -> rejeitado):      {derrubados}")
            print(f"    negativos que sobreviveram:                          {mantidos}")
            print(f"    pessoas PERDIDAS pela térmica (custo do método):      {perdidas}")
            if perdidas:
                print("      ^ este é o número que precisa ficar em zero ou perto. "
                      "Se subir, aumente P_TERMICA_MIN: a térmica está vetando "
                      "vítima, que é o erro mais caro do sistema.")
        else:
            score = np.full(len(razao), np.nan)
    if negativa.sum() == 0:
        razao_neg = np.array([])
        razao_pos = razao[viva & ~np.isnan(razao)]
        score = np.full(len(razao), np.nan)

    resumo_por("vestuário", dados["roupa"][viva], abs_dt_v)
    resumo_por("pose", dados["pose"][viva], abs_dt_v)
    resumo_por("local", dados["local"][viva], abs_dt_v)

    # --- Banda de óbito: proxy de resfriamento -----------------------------
    resf = (classe == "proxy_resfriando") & ~np.isnan(dados["t_marco"])
    if resf.sum() >= 2:
        ordem = np.argsort(dados["t_marco"][resf])
        tm, dtm = dados["t_marco"][resf][ordem], abs_dt[resf][ordem]
        print(f"\nProxy de resfriamento: {resf.sum()} amostras, "
              f"{tm.min():.0f}s a {tm.max():.0f}s, "
              f"|ΔT| de {dtm[0]:.1f}°C para {dtm[-1]:.1f}°C.")
        print("  Use a curva do painel 5 p/ escolher DT_EQUILIBRIO: é o |ΔT| onde "
              "o corpo já se confunde com o fundo. Anotação secundária — não "
              "decide mais aceite de detecção.")
    else:
        print("\n[!] Sem amostras de 'proxy_resfriando': a banda POSSIVEL_OBITO "
              "do detect.py continua sem NENHUM dado por trás.")

    # --- Figura (3x3) ------------------------------------------------------
    # Linha 1: a curva do que uma pessoa produz (calibração do spot-size).
    # Linha 2: sanidade da coleta e triagem secundária.
    # Linha 3: separação positivo/negativo — o que mede o REFORÇO da detecção.
    fig, axs = plt.subplots(3, 3, figsize=(19, 15))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axs.ravel()

    # (1) |ΔT| vs área + envoltória + curva ajustada
    sc = ax1.scatter(area_v, abs_dt_v, c=dist_v, cmap="viridis", s=45,
                     edgecolors="k", linewidths=0.4)
    if len(cen):
        ax1.plot(cen, dt_env, "r--o", linewidth=2,
                 label=f"Envoltória inferior (p{PERCENTIL_ENVOLTORIA})")
    if ajuste is not None:
        grade = np.geomspace(max(1.0, area_v.min()), area_v.max(), 200)
        ax1.plot(grade, modelo_limiar(grade, *ajuste), "b-", linewidth=2,
                 label="Curva ajustada (limiar_dt_vivo)")
    ax1.axvline(AREA_BEM_RESOLVIDO, color="gray", linestyle=":",
                label=f"AREA_BEM_RESOLVIDO={AREA_BEM_RESOLVIDO:.0f}")
    fig.colorbar(sc, ax=ax1, label="Distância (m)")
    ax1.set_xscale("log")
    ax1.set_title("|ΔT| vs. tamanho do bbox (só 'viva')\nborda inferior = limiar de 'vivo'",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Área do bbox (px térmicos, log)", fontsize=11)
    ax1.set_ylabel("|ΔT| = |T_alvo − T_amb| (°C)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9)

    # (2) |ΔT| vs distância (spot-size)
    sc2 = ax2.scatter(dist_v, abs_dt_v, c=area_v, cmap="plasma", s=45,
                      edgecolors="k", linewidths=0.4)
    for d in sorted(set(np.round(dist_v, 1))):
        m = np.round(dist_v, 1) == d
        ax2.plot(d, abs_dt_v[m].mean(), "kX", markersize=10)
    fig.colorbar(sc2, ax=ax2, label="Área bbox (px)")
    ax2.set_title("|ΔT| vs. distância\n(queda = efeito spot-size)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Distância (m)", fontsize=11)
    ax2.set_ylabel("|ΔT| (°C)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # (3) contraste COM SINAL vs T_ambiente -> crossover real
    ax3.scatter(t_amb, dados["dt"], c=dist, cmap="viridis", s=45,
                edgecolors="k", linewidths=0.4)
    ax3.axhline(0, color="k", linewidth=1)
    ax3.axvspan(T_PELE_MIN, T_PELE_MAX, color="orange", alpha=0.15,
                label=f"crossover assumido ({T_PELE_MIN:.0f}-{T_PELE_MAX:.0f}°C)")
    ax3.set_title("Contraste (com sinal) vs. T_ambiente\n(cruza 0 = crossover real)",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("T_ambiente (°C)", fontsize=11)
    ax3.set_ylabel("ΔT com sinal (°C)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=9)

    # (4) sanidade geométrica: área medida vs. prevista pelo IFOV
    ax4.scatter(dist_v, area_v, s=45, c="tab:blue", edgecolors="k", linewidths=0.4,
                label="medido")
    grade_d = np.linspace(max(0.5, dist_v.min()), dist_v.max(), 100)
    ax4.plot(grade_d, [area_esperada_px(d) for d in grade_d], "r-", linewidth=2,
             label="previsto (IFOV, pessoa em pé)")
    ax4.axvline(DISTANCIA_MIN_ENQUADRA_M, color="gray", linestyle=":",
                label=f"enquadra inteiro >{DISTANCIA_MIN_ENQUADRA_M:.1f} m")
    ax4.set_yscale("log")
    ax4.set_title("Área do bbox vs. distância\n(desvio = trena errada ou pose)",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("Distância (m)", fontsize=11)
    ax4.set_ylabel("Área do bbox (px, log)", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=9)

    # (5) proxy de resfriamento -> ancora a banda de óbito
    if resf.sum() >= 2:
        ax5.plot(tm / 60.0, dtm, "o-", color="tab:orange", linewidth=2)
        ax5.set_title("Proxy sem termorregulação\n|ΔT| ao longo do resfriamento",
                      fontsize=12, fontweight="bold")
        ax5.set_xlabel("Tempo desde o marco (min)", fontsize=11)
        ax5.set_ylabel("|ΔT| (°C)", fontsize=11)
        ax5.grid(True, linestyle="--", alpha=0.6)
    else:
        ax5.text(0.5, 0.5, "Sem amostras de\n'proxy_resfriando'\n\n"
                           "A banda POSSIVEL_OBITO\nsegue sem dado.",
                 ha="center", va="center", fontsize=12, color="crimson")
        ax5.set_axis_off()

    # (6) |ΔT| por vestuário — maior fonte de variância da nuvem
    roupas = sorted(set(dados["roupa"][viva]))
    grupos = [abs_dt_v[dados["roupa"][viva] == r] for r in roupas]
    if len(roupas) > 1:
        ax6.boxplot(grupos, tick_labels=roupas)
        ax6.set_title("|ΔT| por vestuário\n(pele exposta domina o contraste)",
                      fontsize=12, fontweight="bold")
        ax6.set_ylabel("|ΔT| (°C)", fontsize=11)
        ax6.grid(True, linestyle="--", alpha=0.6, axis="y")
    else:
        ax6.text(0.5, 0.5, "Só um vestuário coletado.\n\nColete manga curta E\n"
                           "casaco p/ medir o pior caso.",
                 ha="center", va="center", fontsize=12, color="crimson")
        ax6.set_axis_off()

    # (7) razão de contraste: pessoa x não-humano na MESMA régua
    # Em unidades de limiar (1.0 = o contraste esperado de uma pessoa daquele
    # tamanho), então distância não confunde a comparação. Sobreposição das duas
    # nuvens = a térmica não separa, e a fusão não tem o que acrescentar.
    if len(razao_neg):
        ax7.boxplot([razao_pos, razao_neg], tick_labels=["viva", "negativo"],
                    widths=0.5)
        for i, amostra in enumerate([razao_pos, razao_neg], start=1):
            ax7.scatter(np.random.normal(i, 0.06, len(amostra)), amostra,
                        s=25, alpha=0.6,
                        color="tab:green" if i == 1 else "tab:red",
                        edgecolors="k", linewidths=0.3)
        ax7.axhline(1.0, color="tab:blue", linestyle="--",
                    label="razão=1 (contraste pleno de pessoa)")
        ax7.axhline(0.0, color="k", linewidth=1, label="razão=0 (sem contraste)")
        ax7.set_title("Separação pessoa x não-humano\n(régua normalizada pelo limiar)",
                      fontsize=12, fontweight="bold")
        ax7.set_ylabel("ΔT direcionado / limiar do tamanho", fontsize=11)
        ax7.grid(True, linestyle="--", alpha=0.6, axis="y")
        ax7.legend(loc="upper right", fontsize=9)
    else:
        ax7.text(0.5, 0.5, "Sem amostras 'negativo_*'.\n\nSem elas não há taxa de\n"
                           "falso positivo nem\nP_TERMICA_MIN/MAX.",
                 ha="center", va="center", fontsize=12, color="crimson")
        ax7.set_axis_off()

    # (8) o que a fusão MUDA: confiança do YOLO sozinho vs. score fundido
    # Quadrante superior-ESQUERDO = o YOLO perderia (conf<0.5) e o score fundido
    # aceita: resgate. Quadrante inferior-DIREITO = o YOLO aceitaria e a fusão
    # rejeita: queda. Verde no inferior-direito é vítima perdida — o erro caro.
    val = ~np.isnan(score) & ~np.isnan(dados["conf"])
    if val.any():
        for mascara, cor, nome in ((viva & val, "tab:green", "viva"),
                                   (negativa & val, "tab:red", "negativo")):
            if mascara.any():
                ax8.scatter(dados["conf"][mascara], score[mascara], s=45,
                            color=cor, alpha=0.7, edgecolors="k", linewidths=0.4,
                            label=nome)
        ax8.plot([0, 1], [0, 1], "k:", linewidth=1, label="sem efeito da térmica")
        ax8.axvline(CONF_YOLO_SUFICIENTE, color="gray", linestyle="--")
        ax8.axhline(LIMIAR_ACEITE, color="gray", linestyle="--")
        ax8.text(0.03, 0.97, "resgatados\npela térmica", fontsize=9, color="dimgray",
                 transform=ax8.transAxes, va="top")
        ax8.text(0.97, 0.03, "derrubados\npela térmica", fontsize=9, color="dimgray",
                 transform=ax8.transAxes, ha="right")
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.set_title("Efeito da fusão\n(acima da diagonal = térmica reforçou)",
                      fontsize=12, fontweight="bold")
        ax8.set_xlabel("Confiança do YOLO sozinho", fontsize=11)
        ax8.set_ylabel("Score fundido (YOLO + térmica)", fontsize=11)
        ax8.grid(True, linestyle="--", alpha=0.6)
        # Nenhuma amostra tem conf abaixo do portão do YOLO, então a faixa da
        # esquerda está sempre livre para a legenda.
        ax8.legend(loc="center left", fontsize=9)
    else:
        ax8.text(0.5, 0.5, "Sem 'conf' nas amostras\nou sem negativos.\n\n"
                           "Recolete com a versão\natual do script.",
                 ha="center", va="center", fontsize=12, color="crimson")
        ax8.set_axis_off()

    # (9) |ΔT| vs área com os negativos sobrepostos à envoltória de pessoa
    # Negativo ABAIXO do limiar é caso que a fusão resolve. Negativo POUCO acima
    # é confusão real, que nem a fusão nem o veto pegam. Negativo MUITO acima é
    # fonte de calor: sai pelo veto físico, não pela fusão — por isso ele
    # aparecer alto aqui não é sinal de problema.
    ax9.scatter(area_v, abs_dt_v, s=40, color="tab:green", alpha=0.65,
                edgecolors="k", linewidths=0.3, label="viva")
    if negativa.any():
        ax9.scatter(area[negativa], abs_dt[negativa], s=40, color="tab:red",
                    alpha=0.65, marker="^", edgecolors="k", linewidths=0.3,
                    label="negativo")
    if ajuste is not None:
        grade = np.geomspace(max(1.0, area.min()), area.max(), 200)
        ax9.plot(grade, modelo_limiar(grade, *ajuste), "b-", linewidth=2,
                 label="limiar calibrado")
    ax9.set_xscale("log")
    ax9.set_title("Positivos e negativos na curva\n(negativo POUCO acima do limiar = "
                  "confusão real; muito acima = fonte de calor, cai no veto)",
                  fontsize=11, fontweight="bold")
    ax9.set_xlabel("Área do bbox (px térmicos, log)", fontsize=11)
    ax9.set_ylabel("|ΔT| (°C)", fontsize=11)
    ax9.grid(True, linestyle="--", alpha=0.6)
    ax9.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_DADOS, "curva_calibracao.png"), dpi=120)
    print(f"\nGráfico salvo em {os.path.join(PASTA_DADOS, 'curva_calibracao.png')}")
    plt.show()


if __name__ == "__main__":
    main()
