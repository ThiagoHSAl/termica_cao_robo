"""Base comum de geometria e termometria da Thermal Master P1.

Fica aqui tudo que o detect.py e o calibracao_temperatura.py precisam calcular
do MESMO jeito. Se a temperatura ambiente ou o mapeamento área<->distância forem
computados de forma diferente na coleta e na inferência, os limiares calibrados
não valem para o detector — foi por isso que isto virou módulo em vez de ficar
duplicado nos dois scripts.

Contém três blocos:
  - GEOMETRIA/TERMOMETRIA: área<->distância, bbox->térmico, temperatura ambiente;
  - LIMIAR vs. RESOLUÇÃO: a curva calibrada de contraste mínimo (spot-size);
  - FUSÃO: como a evidência térmica REFORÇA a confiança do YOLO, que é o produto
    principal do detector (ver o cabeçalho do detect.py).
"""

import time

import cv2
import numpy as np

# =============================================================================
# GEOMETRIA DO SENSOR
# =============================================================================
# FOV do P1 (confirmado no levantamento da fusão RGB+térmica). As duas contas
# abaixo batem em 5,67 mrad/px, o que confirma pixel quadrado:
#   horizontal: rad(52)/160 = 5,6725e-3   |   vertical: rad(39)/120 = 5,6723e-3
FOV_H_GRAUS = 52.0
FOV_V_GRAUS = 39.0
LARGURA_TERMICA = 160
ALTURA_TERMICA = 120

IFOV_RAD = np.radians(FOV_H_GRAUS) / LARGURA_TERMICA  # rad por pixel térmico

# Alvo de referência para converter distância <-> área do bbox. Pessoa adulta
# em pé, de frente. Serve para ANCORAR limiares numa distância operacional em
# metros (grandeza que se discute e se mede com trena) em vez de num número de
# pixels escolhido a dedo.
PESSOA_ALTURA_M = 1.70
PESSOA_LARGURA_M = 0.45

# Abaixo desta distância a pessoa em pé não cabe na vertical do frame: o bbox
# passa a ser tronco/cabeça e a ÁREA deixa de ser proxy limpo de distância.
DISTANCIA_MIN_ENQUADRA_M = PESSOA_ALTURA_M / (ALTURA_TERMICA * IFOV_RAD)  # ~2,5 m


def area_esperada_px(distancia_m):
    """Área do bbox (px térmicos) de uma pessoa em pé a `distancia_m`.

    Satura nas bordas do frame: de muito perto o corpo é recortado, então a
    área para de crescer com 1/d².
    """
    altura_px = min(ALTURA_TERMICA, PESSOA_ALTURA_M / (distancia_m * IFOV_RAD))
    largura_px = min(LARGURA_TERMICA, PESSOA_LARGURA_M / (distancia_m * IFOV_RAD))
    return float(altura_px * largura_px)


def distancia_estimada_m(area_px):
    """Inverso de `area_esperada_px` — distância implícita numa área de bbox.

    Só vale na região NÃO saturada (d > DISTANCIA_MIN_ENQUADRA_M). Serve para
    rotular gráficos e sanidade da coleta, não para decisão em missão.
    """
    if area_px <= 0:
        return float("inf")
    area_real_m2 = PESSOA_ALTURA_M * PESSOA_LARGURA_M
    return float(np.sqrt(area_real_m2 / (area_px * IFOV_RAD ** 2)))


# =============================================================================
# MAPEAMENTO EXIBIÇÃO -> TÉRMICO
# =============================================================================
def bbox_para_termico(x1, y1, x2, y2, zoom, shape_termico):
    """Converte bbox das coords de exibição para as coords térmicas, com clamp.

    Devolve (tx1, ty1, tx2, ty2) sempre com pelo menos 1 px de lado.
    """
    th, tw = shape_termico
    tx1 = max(0, min(tw - 1, int(x1 / zoom)))
    ty1 = max(0, min(th - 1, int(y1 / zoom)))
    tx2 = max(tx1 + 1, min(tw, int(x2 / zoom)))
    ty2 = max(ty1 + 1, min(th, int(y2 / zoom)))
    return tx1, ty1, tx2, ty2


# =============================================================================
# TEMPERATURA AMBIENTE
# =============================================================================
# Margem de dilatação da máscara: a assinatura térmica da pessoa VAZA para fora
# da caixa do YOLO (borramento por spot-size, cabelo, calor refletido no piso).
# Sem essa folga, o "fundo" ainda contém pixels contaminados pelo alvo.
MARGEM_MASCARA_PX = 3
FRACAO_FUNDO_MINIMA = 0.20  # abaixo disso não sobra fundo suficiente p/ estimar


def temperatura_ambiente(celsius, caixas_termicas=(), margem_px=MARGEM_MASCARA_PX,
                         fracao_minima=FRACAO_FUNDO_MINIMA):
    """Temperatura do fundo do cômodo: mediana dos pixels FORA dos alvos.

    Antes isto era a mediana do frame inteiro, que quebra de perto: a 1-2 m a
    pessoa ocupa metade do frame, a mediana vira a própria pessoa e o ΔT colapsa
    artificialmente — justo nas amostras que deveriam ter o MAIOR contraste.

    Mascarando os bboxes o alvo já sai da conta, então a mediana volta a ser o
    estimador certo do fundo (robusta e sem viés). Um percentil baixo aqui só
    puxaria a referência para a janela/piso mais frio do cômodo.

    Retorna (t_ambiente, confiavel). `confiavel=False` quando sobrou fundo de
    menos e o valor caiu no fallback do frame inteiro — quem chama deve avisar
    o operador em vez de tratar o número como bom.
    """
    if len(caixas_termicas) == 0:
        return float(np.median(celsius)), True

    h, w = celsius.shape
    mascara_fundo = np.ones((h, w), dtype=bool)
    for tx1, ty1, tx2, ty2 in caixas_termicas:
        mascara_fundo[max(0, ty1 - margem_px):min(h, ty2 + margem_px),
                      max(0, tx1 - margem_px):min(w, tx2 + margem_px)] = False

    fundo = celsius[mascara_fundo]
    if fundo.size < fracao_minima * celsius.size:
        return float(np.median(celsius)), False
    return float(np.median(fundo)), True


# =============================================================================
# LIMIAR DE CONTRASTE vs. RESOLUÇÃO DO ALVO
# =============================================================================
def limiar_dt_vivo(area_px, dt_perto, dt_longe, area_ref, expoente):
    """|ΔT| mínimo exigido para "vivo", contínuo na área do bbox.

    Substitui o degrau de dois níveis, que criava um salto de classificação numa
    fronteira onde a física é gradual: uma pessoa a 4,4 m e outra a 4,6 m caíam
    em bandas diferentes por 1 px de diferença de caixa.

    Forma: interpola de `dt_longe` (alvo diluído) até `dt_perto` (alvo cheio),
    saturando em `area_ref`. O expoente < 1 faz a exigência subir rápido logo que
    o alvo começa a resolver e achatar perto da saturação, que é o comportamento
    do preenchimento de pixel (spot-size).
    """
    razao = min(1.0, max(0.0, area_px / area_ref)) ** expoente
    return dt_longe + (dt_perto - dt_longe) * razao


# =============================================================================
# FUSÃO: A TÉRMICA COMO SEGUNDO VOTO NA DETECÇÃO
# =============================================================================
# O detector não usa a térmica para rotular uma detecção já aceita — usa para
# DECIDIR se ela é aceita. Isso exige converter a assinatura térmica em algo
# que se combine com a confiança do YOLO, e não numa classe.

SINAL_ESPERADO = {"frio": +1.0, "quente": -1.0, "crossover": 0.0}


def contraste_direcionado(dt, regime):
    """ΔT projetado na direção em que a PESSOA deve estar, dado o regime.

    Positivo = contraste no sentido esperado (mais quente que o fundo num cômodo
    frio, mais frio num cômodo quente). Assim a mesma curva de limiar serve para
    os dois regimes sem ramificação. No crossover devolve 0: não há direção
    esperada, e a térmica não tem o que dizer.
    """
    return dt * SINAL_ESPERADO.get(regime, 0.0)


def suavizar(x):
    """Smoothstep 3x²−2x³ recortado em [0,1].

    Usado no lugar de degraus para que uma diferença de um décimo de grau perto
    da fronteira não vire uma mudança de decisão — o mesmo motivo pelo qual o
    limiar de contraste virou curva contínua em vez de dois níveis.
    """
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def plausibilidade_termica(dt_direcionado, limiar, p_min, p_max, fator=1.0):
    """P(é pessoa | assinatura térmica), em [p_min, p_max]. 0.5 = abstenção.

    Sobe de `p_min` (nenhum contraste no sentido esperado) até `p_max` (contraste
    pleno, no nível que a calibração diz que uma pessoa produz naquele tamanho de
    alvo).

    Os dois extremos são deliberadamente FECHADOS longe de 0 e de 1:

      - `p_min` > 0 porque ausência de contraste NÃO prova ausência de pessoa. Um
        corpo em equilíbrio térmico com o ambiente lê ΔT~0 e continua sendo uma
        vítima. A térmica atenua a confiança nesse caso, mas nunca veta sozinha.
      - `p_max` < 1 porque contraste presente também não prova pessoa: um animal,
        um motor morno ou um vão de porta com ar quente produzem assinatura
        parecida. A térmica reforça, não decide.

    `fator` em [0,1] é a pertinência à BANDA ABSOLUTA de superfície humana (ver
    plausibilidade_banda). Entra multiplicando a rampa, não o resultado: com
    fator=0 a plausibilidade cai a `p_min`, e NÃO a zero. Isso preserva a regra
    da casa — a térmica atenua, nunca veta sozinha uma caixa do YOLO.

    A única exclusão dura do sistema é a de FONTE DE CALOR, que fica fora desta
    função (ver detect.py): lá o argumento é físico e determinístico — pele
    humana não passa do teto fisiológico, e corpo passivo não ultrapassa o
    ambiente — e não uma pesagem de evidência.
    """
    if limiar <= 0:
        rampa = 1.0 if dt_direcionado > 0 else 0.0
    else:
        rampa = suavizar(dt_direcionado / limiar)
    return p_min + (p_max - p_min) * rampa * min(1.0, max(0.0, float(fator)))


def fundir_confianca(conf_yolo, plausibilidade, eps=1e-6):
    """Atualiza a confiança do YOLO com a evidência térmica, em razão de chances.

        chances_finais = chances_yolo × (p / (1 − p))

    Escolhida por ter o comportamento certo nos três casos que importam:

      - p = 0.5  -> fator 1: a térmica se abstém e o score é a confiança do YOLO
                    intacta. É o que acontece no crossover, onde nenhum alvo tem
                    contraste — e é por isso que ali o sistema não perde nada.
      - p > 0.5  -> multiplica: RESGATA a detecção fraca que o YOLO sozinho
                    descartaria (vítima encolhida, parcialmente coberta).
      - p < 0.5  -> divide: DERRUBA a caixa com forma convincente e assinatura
                    incompatível (manequim, casaco pendurado, reflexo).

    A assimetria de um sistema de resgate está embutida na faixa de `p`: como
    `p_min` é próximo de 0.5 e `p_max` é bem acima, a térmica resgata com mais
    força do que derruba. Uma detecção do YOLO com confiança alta sobrevive à
    térmica adversa; uma com confiança baixa depende dela.
    """
    c = min(max(float(conf_yolo), eps), 1.0 - eps)
    p = min(max(float(plausibilidade), eps), 1.0 - eps)
    chances = (c / (1.0 - c)) * (p / (1.0 - p))
    return chances / (1.0 + chances)


# --- BANDA ABSOLUTA DE SUPERFÍCIE HUMANA -------------------------------------
# O ΔT sozinho não fecha a conta. "3 °C acima do fundo" é satisfeito por parede
# ensolarada, notebook, tomada, o próprio robô — e num cômodo a 20 °C uma pessoa
# NÃO lê 3 °C acima do fundo, lê 10-14 °C acima (o percentil alto pega o rosto).
# Um limiar de contraste fixo é frouxo demais em cômodo frio e apertado demais
# em cômodo morno, porque a grandeza que a física fixa é a temperatura ABSOLUTA
# da superfície humana, não o contraste.
#
# Aqui o contraste medido é reconvertido na temperatura que o alvo TERIA se
# preenchesse o pixel, e essa temperatura é testada contra a banda humana. O
# efeito prático é que a exigência de contraste passa a escalar sozinha com o
# ambiente: cômodo a 20 °C exige muito, cômodo a 30 °C exige pouco, incêndio a
# 45 °C exige contraste negativo — sem nenhuma ramificação por regime.


def fracao_resolvida(limiar_dt, dt_perto):
    """Fração do alvo dentro do pixel, deduzida da própria curva de spot-size.

    A curva `limiar_dt_vivo` já É o contraste cheio multiplicado pelo quanto o
    alvo preenche o pixel naquele tamanho — então a razão entre ela e o valor de
    alvo cheio devolve o fator de diluição, sem precisar de parâmetro novo.

    Por ser uma RAZÃO, ela sobrevive a `dt_perto`/`dt_longe` estarem mal
    calibrados em valor absoluto: só depende da proporção entre os dois e do
    expoente. Domínio (0, 1].
    """
    if dt_perto <= 0:
        return 1.0
    return min(1.0, max(1e-3, limiar_dt / dt_perto))


def temperatura_implicita(t_alvo, t_ambiente, fracao):
    """Temperatura que o alvo teria se preenchesse o pixel (desfaz a diluição).

    O pixel mede uma mistura `t_alvo = f*t_real + (1-f)*t_ambiente`; isto é o
    inverso. Vale nos dois regimes: no quente o ΔT é negativo e a corrigida cai
    ABAIXO do ambiente, que é onde a pessoa deve estar.

    Cuidado ao ler o resultado de alvo pequeno: dividir por `f` amplifica também
    o ruído. Com f=0.33, ±0.5 °C de ruído vira ±1.5 °C aqui — é o preço de
    extrapolar, e o motivo de a banda ter margens generosas.
    """
    return t_ambiente + (t_alvo - t_ambiente) / max(fracao, 1e-3)


def plausibilidade_banda(t_implicita, t_min, t_max, margem):
    """Pertinência [0,1] à banda humana, com ombros suaves de `margem` graus.

    1 dentro de [t_min, t_max], caindo a 0 ao longo de `margem` para fora. Suave
    pelo mesmo motivo dos outros limiares do módulo: 0.1 °C perto da fronteira
    não pode virar mudança de decisão, ainda mais numa grandeza ABSOLUTA, que
    carrega o erro de emissividade e a deriva entre NUCs.
    """
    if margem <= 0:
        return 1.0 if t_min <= t_implicita <= t_max else 0.0
    sobe = suavizar((t_implicita - (t_min - margem)) / margem)
    desce = suavizar(((t_max + margem) - t_implicita) / margem)
    return sobe * desce


def semente_banda(t_ambiente, regime, t_min, t_max, dt_longe, fracao_min):
    """Limiar de segmentação implicado pela banda, no alvo mais diluído.

    Substitui a semente fixa de 1 °C, que era a causa da enxurrada de blobs:
    num cômodo a 20 °C ela acendia toda parede meio grau mais quente que a
    mediana, e o blob resultante ainda vazava para a vizinhança morna, inflando
    a área e estragando os filtros de aspecto e preenchimento.

    Pergunta respondida aqui: qual o MENOR contraste que uma pessoa no limite
    frio da banda, vista o mais longe possível, ainda produziria? Abaixo disso
    não há o que segmentar. Nunca desce abaixo de `dt_longe`, então em ambiente
    morno o comportamento é o de antes.
    """
    if regime == "frio":
        exigido = t_min - t_ambiente
    elif regime == "quente":
        exigido = t_ambiente - t_max
    else:
        return dt_longe
    return max(dt_longe, fracao_min * exigido)


# --- Propostas térmicas ------------------------------------------------------
# Enquanto a térmica só filtrar as caixas do YOLO, ela nunca pode ACRESCENTAR
# uma detecção — no máximo tirar. Quem o YOLO não viu continua invisível, e é
# justamente esse o caso da vítima soterrada, encolhida ou de costas, que não se
# parece com nada do conjunto de treino. Estas propostas são o caminho de volta:
# blobs térmicos compatíveis com pessoa e fora de qualquer caixa do YOLO.
#
# Elas NÃO são detecções. Entram na tela como candidatos a inspecionar de perto.

AREA_MIN_PROPOSTA = 30.0     # px² — abaixo disso é ruído. ~cabeça+ombro a 13 m
FRACAO_MAX_PROPOSTA = 0.35   # blob acima disso é parede/piso aquecido, não corpo
ASPECTO_MAX_PROPOSTA = 5.0   # pessoa em pé é ~3.8:1; deitada, ~1:3.8. Cano é 20:1
PREENCHIMENTO_MIN = 0.35     # área do blob / área do bbox. Corta faixa diagonal
MAX_PROPOSTAS = 5            # teto por frame, das maiores para as menores


def _fracao_coberta(caixa, outras):
    """Fração da área de `caixa` já coberta por alguma das `outras`."""
    x1, y1, x2, y2 = caixa
    area = max(1, (x2 - x1) * (y2 - y1))
    for ox1, oy1, ox2, oy2 in outras:
        iw = max(0, min(x2, ox2) - max(x1, ox1))
        ih = max(0, min(y2, oy2) - max(y1, oy1))
        if iw * ih / area > 0.3:
            return True
    return False


def propor_anomalias_termicas(celsius, t_ambiente, regime, dt_semente,
                              caixas_ocupadas=(), area_min=AREA_MIN_PROPOSTA,
                              fracao_max=FRACAO_MAX_PROPOSTA,
                              aspecto_max=ASPECTO_MAX_PROPOSTA,
                              preenchimento_min=PREENCHIMENTO_MIN,
                              max_propostas=MAX_PROPOSTAS):
    """Bboxes térmicos de anomalias compatíveis com pessoa fora das `caixas_ocupadas`.

    `dt_semente` é só o limiar de SEGMENTAÇÃO — deliberadamente frouxo, o menor
    contraste que a calibração atribui a uma pessoa (alvo distante e diluído).
    Quem aperta é o chamador, aplicando ao blob a MESMA curva de limiar e o mesmo
    veto de fonte de calor que aplica às caixas do YOLO: um blob precisa cumprir
    a exigência de contraste do seu próprio tamanho para virar candidato.

    Os filtros geométricos aqui separam corpo de infraestrutura quente: cano e
    faixa de sol no piso reprovam no aspecto, parede aquecida reprova na área,
    contorno de porta reprova no preenchimento.

    No crossover devolve lista vazia: sem direção esperada de contraste não há o
    que segmentar.
    """
    sinal = SINAL_ESPERADO.get(regime, 0.0)
    if sinal == 0.0:
        return []

    mapa = (celsius - t_ambiente) * sinal
    mascara = (mapa >= dt_semente).astype(np.uint8)
    if not mascara.any():
        return []

    # Abre para matar pixel quente isolado (ruído do sensor), fecha para não
    # partir um corpo em dois por causa de uma faixa de roupa mais fria.
    nucleo = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    area_max = fracao_max * celsius.size

    propostas = []
    for i in range(1, n):
        x, y, w, h, area_blob = stats[i]
        area_bbox = float(w * h)
        if not (area_min <= area_bbox <= area_max):
            continue
        if max(w / max(h, 1), h / max(w, 1)) > aspecto_max:
            continue
        if area_blob / area_bbox < preenchimento_min:
            continue
        caixa = (int(x), int(y), int(x + w), int(y + h))
        if _fracao_coberta(caixa, caixas_ocupadas):
            continue
        propostas.append((area_bbox, caixa))

    propostas.sort(key=lambda p: -p[0])
    return [caixa for _, caixa in propostas[:max_propostas]]


# =============================================================================
# CONTROLE DE NUC / SHUTTER
# =============================================================================
class RelogioNuc:
    """Rastreia o tempo desde o último shutter (NUC) disparado por nós.

    A leitura radiométrica DERIVA entre NUCs, e a câmera dispara o shutter
    sozinha a cada ~90 s. Se a deriva não for controlada, ela entra na amostra
    disfarçada de efeito de distância — e a calibração passa a medir as duas
    coisas somadas.

    Só sabemos a hora dos disparos MANUAIS (o automático não é observável pela
    API), então `segundos()` é um limite superior do tempo desde o último NUC.
    """

    PERIODO_AUTO_S = 90.0

    def __init__(self, limite_alerta_s=60.0):
        self.t_ultimo = None
        self.limite_alerta_s = limite_alerta_s

    def disparar(self, camera):
        """Força o shutter e zera o relógio. Devolve False se a leitura falhou.

        A biblioteca ENVIA o comando de shutter antes de ler o frame de resposta,
        então uma falha na leitura não quer dizer que o NUC deixou de acontecer —
        por isso o relógio é zerado nos dois casos. No pior caso disparamos um
        shutter a mais, o que é barato.

        A exceção não sobe porque derrubar o detector inteiro por causa de um
        frame malformado é pior que perder o frame: quem está rodando isto é um
        robô de busca em operação, e o loop precisa continuar.
        """
        try:
            camera.trigger_shutter()
            ok = True
        except Exception as erro:  # a lib levanta ValueError/USBError conforme o caso
            print(f"AVISO: NUC disparado, mas a leitura do frame falhou ({erro}).")
            ok = False
        self.t_ultimo = time.monotonic()
        return ok

    def segundos(self):
        """Segundos desde o último NUC manual, ou None se nunca houve."""
        if self.t_ultimo is None:
            return None
        return time.monotonic() - self.t_ultimo

    def envelhecido(self):
        """True se convém disparar o shutter antes de capturar."""
        s = self.segundos()
        return s is None or s > self.limite_alerta_s
