"""Calibração da assinatura térmica de pessoa vs. distância (Thermal Master P1).

Objetivo: levantar a curva EMPÍRICA que o detect.py precisa —
    |ΔT| (contraste do alvo contra o fundo)  vs.  área do bbox em px térmicos —
para ESTA câmera, cobrindo o efeito spot-size (a temperatura lida cai com a
distância porque o alvo passa a ocupar menos pixels).

Como usar
---------
1. Informe o cômodo na linha de comando:  --local sala_fria
2. Ajuste a distância na tela com '+' / '-' (passo 0.5 m) até bater com a
   trena, e marque pose ('p') e vestuário ('v') do alvo.
3. 's' dispara o NUC/shutter — faça isso ANTES de cada bloco de capturas.
4. 'c' captura uma amostra (registra a maior detecção de pessoa).
   Capture ~10 amostras por distância, variando pose e ângulo.
5. Repita para 1, 2, 3, 4, 5, 6, 8, 10, 12 m. Os pontos DISTANTES são os que
   mais importam e os mais chatos de coletar — sem eles o DT_VIVO_LONGE sai
   otimista e o detector perde vítima longe.
6. Refaça num cômodo QUENTE (>36°C) e num FRIO (~20°C).
7. 'q' para sair. Dados em calibracao_termica/amostras.csv; cada amostra guarda
   o frame bruto .npy para reanálise.

Classes de alvo ('k' alterna)
-----------------------------
POSITIVOS — definem o contraste que uma pessoa PRODUZ:
  viva              — pessoa viva termorregulando (o caso base)

NEGATIVOS — definem o contraste que um NÃO-humano produz. São eles que medem a
a capacidade da térmica de DERRUBAR falso positivo do YOLO. Aponte a câmera para
o objeto e capture quando o YOLO desenhar caixa nele:
  negativo_manequim — manequim, boneco, casaco pendurado, mochila encostada
  negativo_fonte    — radiador, fogão, notebook, cano quente, luminária
  negativo_reflexo  — pessoa refletida em vidro, espelho ou inox
  negativo_outro    — qualquer outra caixa do YOLO sobre coisa que não é gente

TRIAGEM (secundária) — sustentam a anotação de estado térmico do detect.py:
  proxy_resfriando  — corpo SEM termorregulação esfriando (recipiente com água
                      a ~35°C). 'r' zera o cronômetro no início do resfriamento
                      e cada captura grava os segundos decorridos, levantando a
                      TRAJETÓRIA de resfriamento.
  proxy_ambiente    — objeto já em equilíbrio térmico com o cômodo.

Sem os NEGATIVOS não existe número de falso positivo para apresentar, e a faixa
de plausibilidade do detect.py (P_TERMICA_MIN/MAX) continua sendo chute. Colete
tantos negativos quanto positivos: a estimativa da plausibilidade supõe as duas
classes igualmente representadas.

Depois: rode analisar_calibracao.py, que ajusta a curva e imprime os
parâmetros prontos para colar no topo do detect.py.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from p3_camera import P3Camera, get_model_config, raw_to_celsius
from termica_comum import (
    RelogioNuc,
    bbox_para_termico,
    distancia_estimada_m,
    temperatura_ambiente,
)

# --- CONFIGURAÇÃO ---
MODELO = "thermal_person_finetune_rostos/weights/best.pt"
ZOOM = 4
# MESMO portão do detect.py (CONF_BRUTO), não o limiar de aceite. A amostra
# precisa cobrir a população que o detector realmente vê: se coletássemos a 0.5 e
# o detector opera a partir de 0.2, os parâmetros da fusão sairiam estimados só
# sobre as detecções fortes — justo as que menos dependem da térmica.
CONF = 0.20
# Gravamos AS DUAS caudas da região sem decidir o regime na coleta:
#  - p99 = cauda quente (pessoa em cômodo frio) | p01 = cauda fria (cômodo quente)
# Quem escolhe a cauda certa é a análise. Não decidir aqui evita coletar dado
# inválido num cômodo quente (onde p99 pegaria a parede, não a pessoa fria).
PERCENTIL_QUENTE = 99
PERCENTIL_FRIO = 1
PASSO_DISTANCIA = 0.5         # metros por tecla +/-
NOME_JANELA = "CALIBRACAO_TERMICA"

# Metadados alternáveis por tecla. Sem eles a nuvem de pontos tem variância que
# não dá para explicar depois: vestuário muda o |ΔT| mais que qualquer outro
# fator (define quanta pele exposta entra no bbox) e pose muda a área projetada
# — uma vítima deitada não tem o mesmo bbox de uma pessoa em pé à mesma distância.
CLASSES = ["viva",
           "negativo_manequim", "negativo_fonte", "negativo_reflexo",
           "negativo_outro",
           "proxy_resfriando", "proxy_ambiente"]
POSES = ["em_pe", "sentado", "agachado", "deitado"]
ROUPAS = ["manga_curta", "manga_longa", "casaco"]

COLUNAS = [
    "timestamp", "local", "classe", "pose", "roupa",
    "distancia_m", "area_px_termico", "bbox_w_px", "bbox_h_px",
    "t_p99_c", "t_p01_c", "t_media_c", "t_ambiente_c", "amb_confiavel",
    "t_desde_marco_s", "t_desde_nuc_s", "conf", "arquivo_npy",
]


def maior_deteccao(boxes):
    """Retorna a caixa (xyxy, conf) de maior área, ou None."""
    melhor, melhor_area = None, 0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > melhor_area:
            melhor_area, melhor = area, (x1, y1, x2, y2, float(box.conf[0]))
    return melhor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True,
                    help="identificador do cômodo (ex.: sala_fria, galpao_quente). "
                         "Vai em cada linha do CSV p/ separar as sessões na análise.")
    ap.add_argument("--pasta", default="calibracao_termica")
    ap.add_argument("--distancia-inicial", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.pasta, exist_ok=True)
    arquivo_csv = os.path.join(args.pasta, "amostras.csv")

    model = YOLO(MODELO)
    camera = P3Camera(config=get_model_config("p1"))
    camera.connect()
    camera.start_streaming()

    # Cabeçalho (só se ainda não existe, p/ acumular sessões).
    if not os.path.exists(arquivo_csv):
        with open(arquivo_csv, "w", newline="") as f:
            csv.writer(f, delimiter=";").writerow(COLUNAS)

    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
    distancia_atual = args.distancia_inicial
    n_amostras = 0
    i_classe = i_pose = i_roupa = 0
    t_marco = None          # cronômetro do resfriamento do proxy ('r' zera)
    nuc = RelogioNuc()

    print("Teclas:  '+'/'-' distancia | 'c' captura | 'k' classe | 'p' pose | "
          "'v' vestuario | 's' NUC/shutter | 'r' zera cronometro | 'q' sai")

    try:
        while True:
            ir_brightness, thermal = camera.read_frame_both()
            if ir_brightness is None or thermal is None:
                continue

            celsius = raw_to_celsius(thermal)

            frame = cv2.cvtColor(ir_brightness, cv2.COLOR_GRAY2BGR)
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (w * ZOOM, h * ZOOM),
                               interpolation=cv2.INTER_LINEAR)

            results = model(frame, conf=CONF, classes=0, verbose=False)
            alvo = maior_deteccao(results[0].boxes)

            # Medidas da maior detecção. A caixa precisa ser mapeada ANTES de
            # estimar o ambiente, para poder ser excluída do fundo.
            medida = None
            caixas_termicas = []
            if alvo is not None:
                x1, y1, x2, y2, conf = alvo
                tx1, ty1, tx2, ty2 = bbox_para_termico(x1, y1, x2, y2, ZOOM,
                                                       celsius.shape)
                caixas_termicas.append((tx1, ty1, tx2, ty2))

            # Ambiente = mediana do fundo EXCLUINDO o alvo (ver termica_comum).
            t_ambiente, amb_confiavel = temperatura_ambiente(celsius, caixas_termicas)

            if alvo is not None:
                regiao = celsius[ty1:ty2, tx1:tx2]
                if regiao.size > 0:
                    t_p99 = float(np.percentile(regiao, PERCENTIL_QUENTE))
                    t_p01 = float(np.percentile(regiao, PERCENTIL_FRIO))
                    t_media = float(np.mean(regiao))
                    area_px = (tx2 - tx1) * (ty2 - ty1)
                    bbox_w, bbox_h = (tx2 - tx1), (ty2 - ty1)
                    # Para o HUD: contraste com sinal = cauda mais distante do
                    # ambiente (não decide regime, só mostra a assinatura
                    # dominante ao operador).
                    dt_q, dt_f = t_p99 - t_ambiente, t_p01 - t_ambiente
                    dt = dt_q if abs(dt_q) >= abs(dt_f) else dt_f
                    t_alvo = t_p99 if abs(dt_q) >= abs(dt_f) else t_p01
                    medida = dict(t_p99=t_p99, t_p01=t_p01, t_media=t_media,
                                  area_px=area_px, bbox_w=bbox_w, bbox_h=bbox_h,
                                  t_alvo=t_alvo, dt=dt, conf=conf)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)

            # --- HUD ---
            s_nuc = nuc.segundos()
            s_marco = None if t_marco is None else time.monotonic() - t_marco
            cv2.putText(frame, f"Distancia: {distancia_atual:.1f} m  (+/-)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ambiente: {t_ambiente:.1f}C | Amostras: {n_amostras}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"[{args.local}] {CLASSES[i_classe]} | "
                               f"{POSES[i_pose]} | {ROUPAS[i_roupa]}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 120), 2, cv2.LINE_AA)
            if medida is not None:
                cv2.putText(frame, f"Alvo: {medida['t_alvo']:.1f}C  dT{medida['dt']:+.1f}  "
                                   f"area:{medida['area_px']}px "
                                   f"(~{distancia_estimada_m(medida['area_px']):.1f}m)",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, "Sem deteccao de pessoa", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            aviso_y = 125
            if not amb_confiavel:
                cv2.putText(frame, "! ALVO OCUPA O FRAME - ambiente nao estimavel",
                            (10, aviso_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255), 2, cv2.LINE_AA)
                aviso_y += 22
            if nuc.envelhecido():
                txt = "NUC nunca disparado" if s_nuc is None else f"NUC ha {s_nuc:.0f}s"
                cv2.putText(frame, f"{txt} - aperte 's' antes de capturar",
                            (10, aviso_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 200, 255), 2, cv2.LINE_AA)
                aviso_y += 22
            if s_marco is not None:
                cv2.putText(frame, f"Cronometro: {s_marco:.0f}s", (10, aviso_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2,
                            cv2.LINE_AA)

            cv2.imshow(NOME_JANELA, frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord('q'):
                break
            elif tecla in (ord('+'), ord('=')):
                distancia_atual += PASSO_DISTANCIA
            elif tecla in (ord('-'), ord('_')):
                distancia_atual = max(PASSO_DISTANCIA, distancia_atual - PASSO_DISTANCIA)
            elif tecla == ord('k'):
                i_classe = (i_classe + 1) % len(CLASSES)
                print(f"  classe -> {CLASSES[i_classe]}")
            elif tecla == ord('p'):
                i_pose = (i_pose + 1) % len(POSES)
                print(f"  pose -> {POSES[i_pose]}")
            elif tecla == ord('v'):
                i_roupa = (i_roupa + 1) % len(ROUPAS)
                print(f"  roupa -> {ROUPAS[i_roupa]}")
            elif tecla == ord('s'):
                nuc.disparar(camera)
                print("  NUC/shutter disparado.")
            elif tecla == ord('r'):
                t_marco = time.monotonic()
                print("  Cronometro zerado (inicio do resfriamento do proxy).")
            elif tecla == ord('c'):
                if medida is None:
                    print("  [!] Nenhuma pessoa detectada — amostra ignorada.")
                    continue
                if not amb_confiavel:
                    print("  [!] Fundo insuficiente p/ estimar o ambiente — "
                          "afaste-se ou reenquadre. Amostra ignorada.")
                    continue
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                arq_npy = os.path.join(args.pasta, f"raw_{ts}.npy")
                np.save(arq_npy, thermal)  # frame bruto para reanálise
                with open(arquivo_csv, "a", newline="") as f:
                    csv.writer(f, delimiter=";").writerow([
                        ts, args.local, CLASSES[i_classe], POSES[i_pose],
                        ROUPAS[i_roupa],
                        f"{distancia_atual:.1f}", medida['area_px'],
                        medida['bbox_w'], medida['bbox_h'],
                        f"{medida['t_p99']:.2f}", f"{medida['t_p01']:.2f}",
                        f"{medida['t_media']:.2f}", f"{t_ambiente:.2f}",
                        int(amb_confiavel),
                        "" if s_marco is None else f"{s_marco:.1f}",
                        "" if s_nuc is None else f"{s_nuc:.1f}",
                        f"{medida['conf']:.2f}", os.path.basename(arq_npy),
                    ])
                n_amostras += 1
                print(f"  [{n_amostras}] {CLASSES[i_classe]} d={distancia_atual:.1f}m "
                      f"dT{medida['dt']:+.1f}C area={medida['area_px']}px")

            try:
                if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass
    finally:
        camera.stop_streaming()
        camera.disconnect()
        cv2.destroyAllWindows()
        print(f"\nFim. {n_amostras} amostras nesta sessão. Dados em {arquivo_csv}")


if __name__ == "__main__":
    main()
