# Thermal Detector (Android — modo A: captura de tela + overlay)

App Android que roda o YOLO fine-tunado (`best.onnx`, classe `person`) **em cima**
do app oficial **Thermal Master**, desenhando as caixas de detecção numa
sobreposição transparente. Resultado: **uma única tela** = vídeo térmico + detecções.

## Como funciona
1. O app **Thermal Master** (do fabricante) lê a câmera P1 pelo USB e mostra o vídeo.
2. Este app captura a tela (MediaProjection), roda o `best.onnx` (ONNX Runtime) e
   desenha as caixas num **overlay** por cima de tudo.

Nenhum código de USB é necessário — quem fala com a câmera é o app do fabricante.

## Build
1. Abra a pasta `AndroidThermalDetector` no **Android Studio** (Giraffe ou mais novo).
2. Deixe o Gradle sincronizar (baixa o `onnxruntime-android`).
3. O modelo já está em `app/src/main/assets/best.onnx`.
4. Conecte o S21 FE (depuração USB ligada) e clique em **Run**.

Requisitos: Android 10+ (minSdk 29). O S21 FE atende de sobra.

## Uso
1. Abra o **Thermal Master** e deixe o vídeo térmico aparecendo.
2. Abra o **Thermal Detector** e toque em **Iniciar detecção**.
3. Autorize: (a) "sobrepor a outros apps" e (b) "captura de tela".
4. O app volta ao fundo; retorne ao Thermal Master — as caixas aparecem por cima.
5. Para encerrar: abra o Thermal Detector e toque em **Parar** (ou pela notificação).

## Ajustes rápidos
- Confiança / NMS: `YoloOnnx.kt` → `confThreshold` (0.35) e `iouThreshold` (0.45).
  Para busca e salvamento (não perder ninguém), baixe `confThreshold` p/ ~0.20.
- Aceleração por hardware: em `YoloOnnx.kt`, descomente `addNnapi()` para tentar
  usar a NPU/GPU do aparelho (testar — nem todo operador é suportado por NNAPI).

## Limitações conhecidas
- A inferência roda sobre a **tela inteira**. Como o vídeo térmico ocupa só parte
  da tela, o resto (barra de status, botões do app) entra na imagem. Na prática o
  modelo só dispara sobre pessoas, mas dá para melhorar recortando a região do
  vídeo (crop) antes de inferir — ver `YoloOnnx.detect()`.
- O overlay também é capturado pela MediaProjection (as caixas entram no frame
  seguinte). Como caixas não têm forma de pessoa, não geram falsos positivos.
- A imagem térmica é *white-hot* (grayscale). O modelo foi treinado assim, então
  mantenha o Thermal Master numa paleta grayscale/white-hot para bater com o treino.
