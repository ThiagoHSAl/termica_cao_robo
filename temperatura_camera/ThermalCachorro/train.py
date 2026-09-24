from pathlib import Path
from ultralytics import YOLO

# Diretório deste script (torna os caminhos portáveis dentro/fora de container)
ROOT = Path(__file__).resolve().parent

# Fine-tuning a partir do best.pt (já pré-treinado em imagens térmicas da internet).
# Objetivo: adaptar para pessoas de perto capturadas pela câmera térmica própria.
model = YOLO(str(ROOT / "best.pt"))

results = model.train(
    data=str(ROOT / "data.yaml"),

    # --- Regime de fine-tuning (mesma modalidade térmica -> pouco domain shift) ---
    epochs=100,
    patience=30,          # early stop: dataset pequeno converge/overfita rápido
    lr0=0.001,            # LR inicial baixo p/ não destruir o que o best.pt aprendeu
    lrf=0.01,             # LR final = lr0 * lrf
    warmup_epochs=3.0,
    freeze=None,          # sem congelar: câmera nova ainda tem diferença de domínio

    # --- Resolução / batch ---
    imgsz=640,            # térmica close-up (P1 ~640x512); evita upscale desnecessário
    batch=8,
    multi_scale=True,     # robustez a variação de escala
    rect=False,

    # --- Augmentations ajustadas para TÉRMICA close-up ---
    hsv_h=0.0,            # matiz: sem sentido em térmica
    hsv_s=0.0,            # saturação: sem sentido em térmica
    hsv_v=0.4,            # variação de intensidade/brilho: útil (gain térmico)
    fliplr=0.5,           # espelhamento horizontal
    flipud=0.0,           # pessoa raramente aparece de cabeça p/ baixo
    degrees=5.0,          # pequena rotação
    translate=0.1,
    scale=0.5,            # variação de escala (perto/longe)
    erasing=0.2,          # oclusão parcial (robustez)
    mosaic=0.5,           # moderado: alvos são grandes (perto), não pequenos
    close_mosaic=10,      # desliga mosaic nas últimas 10 épocas p/ refinar
    mixup=0.0,            # embaça alvos térmicos -> desligado
    copy_paste=0.0,       # idem

    # --- Infra ---
    project=str(ROOT / "runs"),
    name="thermal_person_finetune",
    device=0,
    workers=8,
    cache="ram",
    amp=True,
    verbose=True,
)

# Exporta o modelo treinado
path = model.export(format="onnx")
print(f"Modelo exportado para: {path}")
