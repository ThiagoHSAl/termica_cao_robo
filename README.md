# Detecção de pessoas em térmica pura para um cão robô de busca e salvamento

Iniciação Científica no [VerLab](https://verlab.dcc.ufmg.br/) (DCC/UFMG), vertente iniciada em
jul/2026. Uma câmera térmica **Thermal Master P1** (160×120, radiométrica, **sem RGB**) acoplada a
um cão robô para busca em ambientes internos. O detector combina duas evidências:

- **forma:** YOLOv12s fine-tunado em imagens térmicas;
- **assinatura térmica:** a temperatura em °C de cada pixel, lida da radiometria da câmera.

As duas se combinam numa atualização em razão de chances. A térmica não é um filtro aplicado
depois do YOLO. Ela entra na decisão de detectar e age em três direções:

1. **Derruba falso positivo:** forma de pessoa sem assinatura compatível (manequim, casaco,
   reflexo) perde confiança.
2. **Resgata falso negativo:** detecções fracas do YOLO (vítima encolhida, parcialmente soterrada,
   em pose fora do treino) voltam quando a assinatura térmica as sustenta.
3. **Propõe candidatos:** anomalias térmicas com tamanho de pessoa, fora de qualquer caixa, viram
   pontos para o robô se aproximar e reavaliar.

**Estado:** o detector funciona; a calibração que dá procedência aos limiares **ainda não foi
coletada**. Todos os limiares térmicos do `detect.py` são provisórios. O plano está em
`temperatura_camera/ROADMAP.md`.

## Estrutura

| Pasta | O quê |
|---|---|
| `temperatura_camera/` | O projeto: detector, calibração, modelos. Detalhes abaixo. |
| `ThermalDataset/` | Capturas brutas da P1 (24/06, 09/07 e 10/07/2026) que, rotuladas no Roboflow, viraram o `ThermalCachorro`. |
| `AndroidThermalDetector/` | App Android que roda o modelo sobre o vídeo do app oficial da câmera (captura de tela + sobreposição). Tem README próprio. |
| `runs/` | Saída de uma predição de teste do Ultralytics. |

### `temperatura_camera/`

| Arquivo | Papel |
|---|---|
| `ROADMAP.md` | **Fonte de verdade**: estado de cada passo, critérios de conclusão e diário de sessões. |
| `detect.py` | Detector em operação (câmera ao vivo). Todos os limiares no topo do arquivo. |
| `termica_comum.py` | Geometria do sensor, termometria, curva de *spot-size*, fusão. Compartilhado entre detecção e calibração: se divergir, a calibração deixa de valer para o detector. |
| `calibracao_temperatura.py` | Coleta de amostras de assinatura térmica × distância (Passo 1). Ainda não foi executado. |
| `analisar_calibracao.py` | Ajusta a curva às amostras e imprime os parâmetros para o `detect.py`. |
| `temperatura.py`, `plotar.py`, `ensaio_bancada/` | Ensaio de bancada de 1 h (temperatura lida ao longo do tempo) e seus gráficos. |
| `distancia.py` | Gráfico de temperatura × distância com dados coletados num corredor. |
| `EXPLICACAO_SISTEMA_TERMICO.txt` | Explicação do sistema para leigos. |
| `ThermalCachorro/` | Dataset do fine-tune: 165 treino / 70 validação, 1 classe (`person`). [Roboflow](https://universe.roboflow.com/yolodrone-kbw1e/thermal-cachorro), CC BY 4.0. |
| `thermal_person_finetune_rostos/` | **Modelo em produção** (usado pelo `detect.py`). |
| `thermal_person_finetune/` | Fine-tune anterior. |
| `yolo12s_thermal2/`, `yolo12s_thermal-3/` | Treinos-base YOLOv12s (mesmo `train.py` do drone). |

## Modelos

| Modelo | Melhor época | mAP50 | mAP50-95 |
|---|---|---|---|
| `thermal_person_finetune_rostos` (produção) | 94 / 100 | 0,917 | 0,703 |
| `thermal_person_finetune` | 85 / 100 | 0,896 | 0,678 |

> **Estes números estão inflados.** O `ThermalCachorro` foi dividido em treino/validação por
> imagem, não por sessão de captura. Quadros quase idênticos da mesma sessão caem dos dois lados,
> e a validação mede memorização. O re-split por sessão, com um conjunto de teste retido, é o
> Passo 2 do roadmap. Não citar estes valores antes dele.

## Como rodar

```bash
# driver da câmera (engenharia reversa): https://github.com/jvdillon/p3-ir-camera
export PYTHONPATH=~/p3-ir-camera:$PYTHONPATH
cd temperatura_camera
python3 detect.py                                    # detecção ao vivo
python3 calibracao_temperatura.py --local <comodo>   # coleta (Passo 1)
python3 analisar_calibracao.py                        # ajuste da curva
```

Dependências: `ultralytics`, `opencv-python`, `numpy`, `scipy` (análise).

## Relação com os outros projetos

- [`drone_sar`](https://github.com/ThiagoHSAl/drone_sar): a pesquisa anterior (drone, RGB +
  térmica, artigo CROS 2026). A P1 e o driver são os mesmos.
- `mapa_termo_semantico` (repositório privado): a direção atual (Pioneer 3-DX + LIDAR + térmica
  em ROS 2). O detector daqui é o que vai alimentar o mapa semântico.
