"""Detecção de pessoas em imagem térmica: YOLO REFORÇADO pela assinatura térmica.

A térmica não é um rótulo aplicado depois que o YOLO já decidiu — ela é um
segundo voto que entra na própria decisão de detectar, nas três direções:

  1. DERRUBA falso positivo. Caixa com forma convincente e sem assinatura
     térmica compatível (manequim, casaco pendurado, reflexo em vidro, textura
     de escombro) tem a confiança atenuada e pode cair abaixo do aceite.

  2. RESGATA falso negativo. O portão do YOLO é aberto (CONF_BRUTO bem abaixo de
     0.5) e as detecções fracas — vítima encolhida, parcialmente soterrada, de
     costas, em pose que não existe no conjunto de treino — voltam a ser aceitas
     quando a assinatura térmica as sustenta. Este é o ganho principal: num
     desabamento é aqui que o YOLO sozinho falha.

  3. PROPÕE o que o YOLO não viu. Anomalias térmicas com tamanho e forma
     compatíveis com pessoa, fora de qualquer caixa, viram CANDIDATOS para o
     robô aproximar e reavaliar. Não contam como detecção.

A triagem térmica (corpo ativo × corpo em equilíbrio com o ambiente) continua
existindo, mas REBAIXADA a anotação de uma caixa já aceita. Ela não decide mais
o que aparece na tela — antes decidia, e por isso a temperatura acabava podendo
apagar uma vítima. Agora o que apaga uma caixa é o score fundido, e o único veto
duro é o de fonte de calor, que é argumento físico e não pesagem de evidência.

A cor da caixa passa a codificar a ORIGEM da detecção, não o estado do corpo:
é ela que mostra, em operação e em vídeo de bancada, o que a térmica acrescentou.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from p3_camera import P3Camera, get_model_config, raw_to_celsius
from termica_comum import (
    RelogioNuc,
    area_esperada_px,
    bbox_para_termico,
    contraste_direcionado,
    distancia_estimada_m,
    fracao_resolvida,
    fundir_confianca,
    limiar_dt_vivo as _limiar_dt_vivo,
    plausibilidade_banda,
    plausibilidade_termica,
    propor_anomalias_termicas,
    semente_banda,
    temperatura_ambiente,
    temperatura_implicita,
)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
MODELO = "thermal_person_finetune_rostos/weights/best.pt"
ZOOM = 4          # fator de ampliação para exibição (P1 é só 160x120)
NMS_IOU = 0.5     # NMS mais apertado que o default (0.7) p/ suprimir caixas
                  # DUPLICADAS sobre a mesma pessoa (viravam detecções sobrepostas).
NOME_JANELA = "RESULTADO_DETECCAO"

# --- PORTÃO DO YOLO E ACEITE FINAL ------------------------------------------
# CONF_BRUTO é o portão do YOLO, não a decisão. Ele é baixo de propósito: acima
# dele o candidato entra na fusão, e quem decide é o score fundido. Manter 0.5
# aqui anularia a direção (2) do reforço — a detecção fraca morreria ANTES de a
# térmica poder opinar, e a térmica só saberia tirar, nunca acrescentar.
#
# LIMIAR_ACEITE é deliberadamente igual ao CONF antigo (0.5): a barra final de
# aceite não mudou, só passou a ser aplicada a uma confiança informada pelas duas
# evidências. Sem isso, qualquer ganho de recall seria só o efeito de baixar o
# limiar, e a comparação com o detector anterior não valeria nada.
CONF_BRUTO = 0.20
LIMIAR_ACEITE = 0.50
CONF_YOLO_SUFICIENTE = 0.50   # acima disso o YOLO já bastava sozinho (p/ contar
                              # quantas detecções são MÉRITO da térmica)

# --- FAIXA DA EVIDÊNCIA TÉRMICA ---------------------------------------------
# Extremos da plausibilidade (ver termica_comum.plausibilidade_termica). São o
# quanto a térmica pode empurrar a decisão em cada sentido, e a assimetria é
# intencional: resgatar vale mais que derrubar num sistema de resgate.
#
#   P_TERMICA_MAX = 0.85 -> chances x5.7  (conf 0.25 vira score 0.65: resgate)
#   P_TERMICA_MIN = 0.35 -> chances x0.54 (conf 0.60 vira score 0.45: queda;
#                                          conf 0.90 vira 0.83: sobrevive)
#
# PROVISÓRIOS como todos os outros: saem da separação medida entre amostras
# 'viva' e amostras 'negativo_*' em analisar_calibracao.py. Até lá, são ordem de
# grandeza plausível — e sem os negativos coletados não há como afirmar taxa de
# falso positivo nenhuma.
P_TERMICA_MIN = 0.35
P_TERMICA_MAX = 0.85
P_TERMICA_NEUTRA = 0.50       # crossover: a térmica se abstém, score = conf YOLO

# --- Banda de superfície de pele humana (teto fisiológico) ------------------
# Um corpo vivo regulando mantém a superfície numa faixa limitada, INDEPENDENTE
# do ambiente: a pele exposta fica ~33-36°C e não passa muito disso (suor até
# a esfria). Essa faixa define os REGIMES térmicos do cômodo:
#   - ambiente ABAIXO de ~T_PELE_MIN  -> pessoa é anomalia QUENTE (ΔT>0)
#   - ambiente ENTRE min e max        -> crossover: contraste some (ΔT~0)
#   - ambiente ACIMA de ~T_PELE_MAX   -> pessoa é anomalia FRIA  (ΔT<0)
T_PELE_MIN = 33.0          # °C — piso da faixa de pele/superfície humana
T_PELE_MAX = 37.0          # °C — teto fisiológico da superfície

# Veto de fonte de calor (radiador, fogão, eletrônico, cano). É a ÚNICA exclusão
# dura que sobrou: não passa pela fusão, elimina a caixa. O critério é físico e
# determinístico, e PRECISA ser diferente em cada regime:
#
#  - REGIME FRIO: vale o TETO ABSOLUTO. Com o cômodo abaixo da pele, nada
#    passivo chega a 40°C; quem está lá em cima gera calor próprio.
#
#  - REGIME QUENTE: o teto absoluto PERDE O SENTIDO — num incêndio a 45°C tudo
#    no cômodo está acima de 40°C, inclusive um corpo que já equilibrou com o
#    ambiente. Aplicar o teto ali vetava a vítima. O que vale no regime quente é
#    o argumento RELATIVO: um corpo passivo no máximo ALCANÇA o ambiente, nunca
#    o ultrapassa. Só é fonte de calor quem lê MAIS QUENTE QUE O PRÓPRIO
#    AMBIENTE por uma margem.
T_VETO_FONTE_CALOR = 41.0  # °C absolutos (só no REGIME FRIO)
DT_VETO_FONTE_CALOR = 5.0  # °C acima do ambiente (só no REGIME QUENTE)

# Contraste (|ΔT|) que uma pessoa produz, dependente de quão resolvido está o
# alvo. Alvo grande (perto) -> contraste cheio. Alvo pequeno (longe) -> valor
# diluído pelo efeito spot-size. É a régua da plausibilidade: o quanto o alvo
# se aproxima desta curva é o quanto ele "parece pessoa" para a térmica.
#
# A fronteira "bem resolvido" é ancorada numa DISTÂNCIA OPERACIONAL e convertida
# em área pela geometria do sensor (IFOV). Antes era um número de pixels solto:
# se a calibração fosse coletada mais de perto ou mais de longe, a mediana das
# áreas mudava e a fronteira andava junto — era artefato de amostragem, não
# física do alvo.
DISTANCIA_BEM_RESOLVIDO_M = 4.5   # a partir daqui o alvo conta como "perto"
AREA_BEM_RESOLVIDO = area_esperada_px(DISTANCIA_BEM_RESOLVIDO_M)  # ~1174 px

DT_VIVO_PERTO = 3.0        # |ΔT| esperado de pessoa com alvo bem resolvido
DT_VIVO_LONGE = 1.0        # |ΔT| esperado de pessoa com alvo pequeno/diluído
EXPOENTE_LIMIAR = 0.5      # curvatura da transição perto<->longe (spot-size)
DT_EQUILIBRIO = 0.5        # |ΔT| abaixo do qual o alvo é indistinguível do fundo

# --- BANDA ABSOLUTA DE SUPERFÍCIE HUMANA ------------------------------------
# A régua de ΔT diz se HÁ contraste; esta banda diz se o contraste tem o TAMANHO
# de gente. Só o ΔT é frouxo demais: num cômodo a 18 °C, "1 °C acima do fundo"
# é parede, tomada, o próprio robô — e uma pessoa ali não lê 19 °C, lê ~33 °C no
# percentil alto. A grandeza que a fisiologia fixa é a temperatura ABSOLUTA da
# superfície, então é ela que tem que ser testada.
#
# O ΔT medido é reconvertido na temperatura que o alvo teria se preenchesse o
# pixel (termica_comum.temperatura_implicita) e comparado com esta faixa.
#
# São DUAS faixas, com propósitos diferentes:
#
#   BANDA HUMANA [T_HUMANO_MIN, T_HUMANO_MAX] — o que uma superfície humana
#   pode ler. Governa o VOTO térmico sobre as caixas do YOLO, e governa
#   SUAVEMENTE: fora dela a plausibilidade cai a P_TERMICA_MIN, nunca a zero,
#   e os ombros de MARGEM_BANDA fazem a queda ser gradual.
#
#   CORTE DO CANDIDATO [T_CANDIDATO_MIN, T_CANDIDATO_MAX] — onde a proposta
#   térmica é aceita ou descartada, e aqui o corte é DURO. É deliberadamente
#   mais LARGO que a banda do voto, e a assimetria tem razão: o erro de esconder
#   pesa mais de um lado que do outro. Esconder um candidato é o robô não ir
#   olhar; atenuar uma caixa do YOLO ainda deixa a forma sustentá-la. Some a
#   isso que a temperatura implícita é uma EXTRAPOLAÇÃO (divide pela fração
#   resolvida) e no alvo pequeno amplifica o ruído — justamente o alvo distante,
#   que é quem mais precisa da folga.
#
# A banda 30-40 do voto é o envelope do que uma pele lida por microbolômetro sem
# refrigeração pode marcar: pele exposta fica em 33-36, e o resto é folga para
# vasoconstrição, rosto parcialmente coberto, erro absoluto do sensor (±2-3 °C)
# e deriva entre NUCs. O corte 25-41 do candidato abre mais 5 °C para baixo, que
# é onde cai a vítima distante depois da extrapolação.
T_HUMANO_MIN = 28.0
T_HUMANO_MAX = 40.0
MARGEM_BANDA = 2.0         # ombros suaves da banda (°C), p/ o voto térmico

T_CANDIDATO_MIN = 25.0     # corte DURO da proposta térmica (mais largo que a banda)
T_CANDIDATO_MAX = 41.0

# Diluição máxima admitida (alvo mais distante da curva calibrada). Sai da
# própria razão da curva de spot-size, então acompanha qualquer recalibração.
FRACAO_MIN_RESOLVIDA = DT_VIVO_LONGE / DT_VIVO_PERTO

# Percentil da região usado como "temperatura do alvo" (robustez a pixel morto).
# No regime QUENTE a pessoa é FRIA, então lá usamos o percentil BAIXO (o pixel
# mais frio é o mais "cheio de pessoa"); no regime frio, o percentil alto.
PERCENTIL_ALVO = 99
PERCENTIL_ALVO_FRIO = 1

# --- Cores BGR por ORIGEM da detecção ---------------------------------------
# A cor deixou de dizer "vivo/óbito" e passou a dizer DE ONDE veio a detecção.
# É o que torna a contribuição da térmica visível na tela e contável em vídeo.
COR_CONFIRMADO = (0, 255, 0)      # verde   — YOLO já bastava, térmica concorda
COR_RESGATE = (0, 220, 255)       # amarelo — YOLO fraco, térmica sustentou
COR_SO_FORMA = (200, 200, 0)      # ciano   — crossover: térmica se absteve
COR_CANDIDATO = (255, 150, 0)     # azul    — proposta térmica, YOLO não viu
COR_REJEITADO = (0, 0, 200)       # vermelho— derrubado (só com DEPURAR=True)

DEPURAR = False   # 'd' — derrubados/vetados/fora da banda, em vermelho

# --- CAMADAS DE DESENHO -----------------------------------------------------
# Três chaves independentes, uma por tipo de caixa, para isolar UMA camada de
# cada vez na tela — em bancada é a única forma de julgar uma delas sem as
# outras por cima. Nenhuma altera a decisão: os contadores do HUD são sempre
# calculados com tudo ligado, porque caixa escondida em silêncio é vítima
# esquecida. Quando uma camada está oculta o HUD marca "(ocultas)".
# Todas começam DESLIGADAS: o padrão é a térmica limpa com o HUD, e o operador
# acende a camada que quer olhar. Os contadores do HUD continuam corretos desde
# o primeiro frame, então nada fica invisível — só não desenhado.
MOSTRAR_DETECCOES = False   # 'y' — caixas aceitas: [Y+T] verde, [T>Y] amarelo, [Y] ciano
MOSTRAR_CANDIDATOS = False  # 'c' — propostas térmicas: [T] CANDIDATO, azul


def limiar_dt_pessoa(area_px_termico):
    """|ΔT| que uma pessoa produz naquele tamanho de bbox (curva calibrada).

    Curva CONTÍNUA (ver termica_comum.limiar_dt_vivo). Os 4 parâmetros saem do
    ajuste feito por analisar_calibracao.py sobre a envoltória inferior das
    amostras de pessoa viva.
    """
    return _limiar_dt_vivo(area_px_termico, DT_VIVO_PERTO, DT_VIVO_LONGE,
                           AREA_BEM_RESOLVIDO, EXPOENTE_LIMIAR)


def temperatura_implicita_alvo(t_alvo, t_ambiente, area_px_termico):
    """Temperatura absoluta que o alvo teria se preenchesse o pixel.

    Desfaz a diluição de spot-size usando a MESMA curva que define o limiar de
    contraste — é o número que a banda humana julga. Num cômodo a 18 °C: parede
    a 19 °C devolve ~19-21 °C (fora da banda), pessoa a 5 m devolve ~33 °C.
    """
    fracao = fracao_resolvida(limiar_dt_pessoa(area_px_termico), DT_VIVO_PERTO)
    return temperatura_implicita(t_alvo, t_ambiente, fracao)


def temperatura_alvo(regiao, regime):
    """Temperatura representativa do alvo, escolhendo a cauda certa da região.

    Regime quente -> a pessoa é o ponto FRIO (percentil baixo).
    Caso contrário -> a pessoa é o ponto QUENTE (percentil alto).
    """
    if regime == "quente":
        return float(np.percentile(regiao, PERCENTIL_ALVO_FRIO))
    return float(np.percentile(regiao, PERCENTIL_ALVO))


def regime_termico(t_ambiente):
    """Classifica o ambiente do cômodo relativo à faixa de pele humana."""
    if t_ambiente < T_PELE_MIN:
        return "frio"      # caso normal: pessoa mais quente que o fundo
    if t_ambiente > T_PELE_MAX:
        return "quente"    # ambiente acima da pele: pessoa mais fria que o fundo
    return "crossover"     # ambiente ~ pele: contraste térmico some


def eh_fonte_calor(t_alvo, dt, regime):
    """Veto físico: o alvo é quente demais para ser um corpo humano?

    Fora da fusão de propósito. Não é uma evidência a ser pesada contra a forma;
    é uma impossibilidade — pele não passa do teto fisiológico, e corpo passivo
    não ultrapassa o ambiente. Uma caixa que reprova aqui não é pessoa, por mais
    convicto que o YOLO esteja.
    """
    if regime == "quente":
        return dt > DT_VETO_FONTE_CALOR
    if regime == "frio":
        return t_alvo > T_VETO_FONTE_CALOR
    # Crossover: o ambiente já está na faixa de pele. O teto absoluto ainda
    # separa (40°C segue acima de qualquer pele), mas com margem menor.
    return t_alvo > T_VETO_FONTE_CALOR


def evidencia_termica(t_alvo, t_ambiente, area_px_termico, regime):
    """Converte a assinatura térmica da região em plausibilidade de ser pessoa.

    São DUAS perguntas, e as duas precisam ser respondidas: existe contraste no
    sentido esperado (régua de ΔT), e esse contraste implica uma temperatura de
    superfície humana (banda absoluta)? A segunda é o que separa pessoa de parede
    morna, porque parede morna TEM contraste — só não tem 33 °C por trás dele.

    A banda entra como FATOR SUAVE, não como veto: fora dela a plausibilidade cai
    a P_TERMICA_MIN e a caixa do YOLO ainda sobrevive se a forma for convincente.
    Vítima sob cobertor lê frio demais para a banda e continua detectável.

    Retorna (plausibilidade, dt, dt_direcionado, t_implicita). No crossover
    devolve a plausibilidade NEUTRA: o ambiente está na temperatura da pele,
    ninguém tem contraste, e pesar essa ausência contra a forma seria descartar
    vítima por um defeito do método. Neutro faz o score fundido ser a confiança
    do YOLO intacta — a térmica cala a boca em vez de atrapalhar.
    """
    dt = t_alvo - t_ambiente
    dt_dir = contraste_direcionado(dt, regime)
    t_impl = temperatura_implicita_alvo(t_alvo, t_ambiente, area_px_termico)
    if regime == "crossover":
        return P_TERMICA_NEUTRA, dt, dt_dir, t_impl
    fator = plausibilidade_banda(t_impl, T_HUMANO_MIN, T_HUMANO_MAX, MARGEM_BANDA)
    plaus = plausibilidade_termica(dt_dir, limiar_dt_pessoa(area_px_termico),
                                   P_TERMICA_MIN, P_TERMICA_MAX, fator)
    return plaus, dt, dt_dir, t_impl


def estado_termico(dt_dir, area_px_termico, regime):
    """Triagem REBAIXADA: anotação sobre uma detecção já aceita.

    Não influencia mais o aceite nem a cor da caixa — é informação extra para a
    equipe priorizar a fila, e nada mais. A distinção real não é vida x morte
    (um corpo leva horas para equilibrar com o ambiente): é corpo em
    DESEQUILÍBRIO térmico x corpo em EQUILÍBRIO. Nunca é motivo de descarte.
    """
    if regime == "crossover":
        return "n/d"                       # sem contraste possível para ninguém
    if dt_dir >= limiar_dt_pessoa(area_px_termico):
        return "ativo"                     # produzindo calor contra o ambiente
    if abs(dt_dir) < DT_EQUILIBRIO:
        if area_px_termico >= AREA_BEM_RESOLVIDO:
            return "equilibrio"            # perto e sem contraste: medida confiável
        return "equilibrio?"               # longe: pode ser só diluição spot-size
    return "fraco"                         # entre as duas: contraste parcial


def desenhar_alvo(frame, caixa_exib, cor, linha_cima, linha_baixo, espessura=2):
    """Caixa + duas linhas de rótulo, em coordenadas de exibição."""
    x1, y1, x2, y2 = (int(v) for v in caixa_exib)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, espessura)
    cv2.putText(frame, linha_cima, (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
    cv2.putText(frame, linha_baixo, (x1, min(frame.shape[0] - 5, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)


# =============================================================================
# INICIALIZAÇÃO
# =============================================================================
model = YOLO(MODELO)

camera = P3Camera(config=get_model_config("p1"))
camera.connect()
camera.start_streaming()

cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.moveWindow(NOME_JANELA, 10, 10)

nuc = RelogioNuc()
print("Teclas:  'y' deteccoes  |  'c' candidatos termicos  |  'd' depuracao  |  "
      "'s' força NUC/shutter  |  'q' sai")

try:
    while True:
        # ir_brightness: imagem 8-bit (AGC do hardware), shape (120, 160)
        # thermal: valor bruto uint16 por pixel (1/64 Kelvin), shape (120, 160)
        ir_brightness, thermal = camera.read_frame_both()
        if ir_brightness is None or thermal is None:
            continue

        # Mapa de temperatura absoluta (°C) por pixel, no espaço térmico 160x120.
        celsius = raw_to_celsius(thermal)

        # Imagem de exibição: térmica white-hot -> BGR, ampliada por ZOOM.
        frame = cv2.cvtColor(ir_brightness, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w * ZOOM, h * ZOOM), interpolation=cv2.INTER_LINEAR)

        # Inferência (só classe 0 = pessoa) com o portão ABERTO em CONF_BRUTO:
        # o que chega aqui é candidato, não detecção. NMS apertado + agnostic_nms
        # suprimem caixas sobrepostas duplicadas sobre a mesma pessoa — passa a
        # importar mais com o portão baixo, que produz mais caixas fracas.
        results = model(frame, conf=CONF_BRUTO, classes=0, iou=NMS_IOU,
                        agnostic_nms=True, verbose=False)

        # --- PASSADA 1: mapear todas as caixas para o espaço térmico ---------
        # Precisa vir ANTES da temperatura ambiente: o fundo do cômodo só é
        # estimável depois de saber quais pixels pertencem a candidatos.
        candidatos = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tcaixa = bbox_para_termico(x1, y1, x2, y2, ZOOM, celsius.shape)
            candidatos.append(((x1, y1, x2, y2), tcaixa, float(box.conf[0])))

        # Temperatura ambiente: mediana do fundo, EXCLUINDO os candidatos (com
        # folga p/ o vazamento térmico ao redor da pessoa).
        t_ambiente, amb_confiavel = temperatura_ambiente(
            celsius, [t for _, t, _ in candidatos])

        # Regime térmico do cômodo (define o sentido esperado do contraste).
        regime = regime_termico(t_ambiente)

        # --- PASSADA 2: fundir forma + térmica em cada candidato -------------
        n_aceito = n_resgate = n_rejeitado = n_vetado = 0
        for (x1, y1, x2, y2), (tx1, ty1, tx2, ty2), conf in candidatos:
            regiao = celsius[ty1:ty2, tx1:tx2]
            if regiao.size == 0:
                continue

            t_alvo = temperatura_alvo(regiao, regime)
            area_px = (tx2 - tx1) * (ty2 - ty1)  # área em pixels térmicos

            # Veto físico (fora da fusão): quente demais para ser corpo humano.
            if eh_fonte_calor(t_alvo, t_alvo - t_ambiente, regime):
                n_vetado += 1
                if DEPURAR:
                    desenhar_alvo(frame, (x1, y1, x2, y2), COR_REJEITADO,
                                  f"VETO fonte de calor {t_alvo:.1f}C",
                                  f"conf {conf:.2f}", espessura=1)
                continue

            plaus, dt, dt_dir, t_impl = evidencia_termica(t_alvo, t_ambiente,
                                                          area_px, regime)
            score = fundir_confianca(conf, plaus)

            if score < LIMIAR_ACEITE:
                # A térmica derrubou (ou a forma era fraca demais para ser
                # sustentada por ela). Só interessa ver isto em depuração.
                if conf >= CONF_YOLO_SUFICIENTE:
                    n_rejeitado += 1   # o YOLO sozinho teria aceitado: a térmica
                                       # é que derrubou. Direção (1) do reforço.
                if DEPURAR:
                    desenhar_alvo(frame, (x1, y1, x2, y2), COR_REJEITADO,
                                  f"REJEITADO {score:.2f}  (yolo {conf:.2f})",
                                  f"dT{dt:+.1f} T~{t_impl:.0f}C p{plaus:.2f}",
                                  espessura=1)
                continue

            n_aceito += 1
            if conf < CONF_YOLO_SUFICIENTE:
                n_resgate += 1         # o YOLO sozinho teria perdido esta vítima.
                cor, origem = COR_RESGATE, "T>Y"
            elif regime == "crossover":
                cor, origem = COR_SO_FORMA, "Y"
            else:
                cor, origem = COR_CONFIRMADO, "Y+T"

            if not MOSTRAR_DETECCOES:
                continue
            estado = estado_termico(dt_dir, area_px, regime)
            desenhar_alvo(frame, (x1, y1, x2, y2), cor,
                          f"[{origem}] {score:.2f}  (yolo {conf:.2f})",
                          f"~{distancia_estimada_m(area_px):.1f}m  dT{dt:+.1f}  "
                          f"T~{t_impl:.0f}C  {estado}")

        # --- PASSADA 3: o que a térmica viu e o YOLO não ---------------------
        # Sem isto a térmica nunca acrescenta detecção, só tira — e a vítima
        # soterrada, que não tem forma de gente nenhuma, fica invisível.
        # A semente sai da BANDA, não de um 1 °C fixo: é o menor contraste que
        # uma pessoa no limite frio da banda, vista o mais longe possível, ainda
        # produziria neste ambiente. Num cômodo a 18 °C isso é ~3.3 °C, e a
        # parede a 19 °C deixa de virar blob antes mesmo de ser filtrada.
        dt_semente = semente_banda(t_ambiente, regime, T_CANDIDATO_MIN,
                                   T_CANDIDATO_MAX, DT_VIVO_LONGE,
                                   FRACAO_MIN_RESOLVIDA)
        propostas = propor_anomalias_termicas(
            celsius, t_ambiente, regime, dt_semente,
            caixas_ocupadas=[t for _, t, _ in candidatos])

        n_candidato = n_fora_banda = 0
        for (tx1, ty1, tx2, ty2) in propostas:
            regiao = celsius[ty1:ty2, tx1:tx2]
            if regiao.size == 0:
                continue
            t_alvo = temperatura_alvo(regiao, regime)
            dt = t_alvo - t_ambiente
            area_px = (tx2 - tx1) * (ty2 - ty1)

            # Mesmo veto e mesma régua de contraste das caixas do YOLO: o blob
            # precisa cumprir a exigência do PRÓPRIO tamanho para virar candidato.
            if eh_fonte_calor(t_alvo, dt, regime):
                continue
            dt_dir = contraste_direcionado(dt, regime)
            if dt_dir < limiar_dt_pessoa(area_px):
                continue

            # Banda humana, aqui DURA — ao contrário do voto sobre a caixa do
            # YOLO, que só é atenuado. A diferença é que a proposta não tem
            # evidência de forma nenhuma para sustentá-la: sem a banda ela é só
            # "mancha morna", e é assim que a tela inunda. Uma proposta que não
            # implica temperatura de gente não vale o deslocamento do robô.
            #
            # O corte usa a faixa do CANDIDATO, mais frouxa que a banda do voto:
            # a temperatura implícita é extrapolada (divide pela fração
            # resolvida), e no alvo pequeno essa divisão amplifica o ruído.
            # Apertar aqui até a banda fisiológica derrubaria a vítima distante.
            t_impl = temperatura_implicita_alvo(t_alvo, t_ambiente, area_px)
            caixa_exib = (tx1 * ZOOM, ty1 * ZOOM, tx2 * ZOOM, ty2 * ZOOM)
            if not (T_CANDIDATO_MIN <= t_impl <= T_CANDIDATO_MAX):
                n_fora_banda += 1
                if DEPURAR:
                    desenhar_alvo(frame, caixa_exib, COR_REJEITADO,
                                  f"FORA DA BANDA T~{t_impl:.0f}C",
                                  f"dT{dt:+.1f}", espessura=1)
                continue

            n_candidato += 1
            if not MOSTRAR_CANDIDATOS:
                continue
            desenhar_alvo(frame, caixa_exib, COR_CANDIDATO,
                          "[T] CANDIDATO - aproximar",
                          f"~{distancia_estimada_m(area_px):.1f}m  dT{dt:+.1f}"
                          f"  T~{t_impl:.0f}C",
                          espessura=1)

        # --- HUD --------------------------------------------------------------
        cv2.putText(frame,
                    f"Ambiente: {t_ambiente:.1f}C  [regime: {regime}]  "
                    f"banda {T_HUMANO_MIN:.0f}-{T_HUMANO_MAX:.0f}C  "
                    f"cand {T_CANDIDATO_MIN:.0f}-{T_CANDIDATO_MAX:.0f}C  "
                    f"semente dT{dt_semente:+.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # Placar do reforço: quanto a térmica ACRESCENTOU e quanto TIROU neste
        # frame. É o número que justifica a fusão existir.
        # Com o desenho desligado o contador permanece — o operador precisa saber
        # que existem candidatos ali, mesmo sem as caixas na tela.
        marca_cand = "" if MOSTRAR_CANDIDATOS else " (ocultos)"
        marca_det = "" if MOSTRAR_DETECCOES else " (ocultas)"
        cv2.putText(frame,
                    f"YOLO:{len(candidatos)}  aceitos:{n_aceito}{marca_det} "
                    f"(resgates:{n_resgate})  derrubados:{n_rejeitado}  "
                    f"vetos:{n_vetado}  cand.T:{n_candidato}{marca_cand}  "
                    f"fora-banda:{n_fora_banda}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1,
                    cv2.LINE_AA)
        # Alvo ocupando quase todo o frame -> o fundo não dá mais para estimar,
        # e é o t_ambiente que decide o REGIME. Avisar em vez de silenciar.
        if not amb_confiavel:
            cv2.putText(frame, "! ALVO OCUPA O FRAME - ambiente/regime incertos",
                        (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                        cv2.LINE_AA)
        if nuc.envelhecido():
            cv2.putText(frame, "NUC antigo ('s' p/ shutter)",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)
        if DEPURAR:
            cv2.putText(frame, "DEPURACAO: mostrando derrubados/vetados",
                        (10, frame.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 200), 1, cv2.LINE_AA)

        cv2.imshow(NOME_JANELA, frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        if tecla == ord('s'):
            nuc.disparar(camera)
        if tecla == ord('d'):
            DEPURAR = not DEPURAR
        if tecla == ord('c'):
            MOSTRAR_CANDIDATOS = not MOSTRAR_CANDIDATOS
        if tecla == ord('y'):
            MOSTRAR_DETECCOES = not MOSTRAR_DETECCOES
        # WND_PROP_VISIBLE depende do backend Qt e falha em alguns builds.
        try:
            if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            pass
finally:
    camera.stop_streaming()
    camera.disconnect()
    cv2.destroyAllWindows()
