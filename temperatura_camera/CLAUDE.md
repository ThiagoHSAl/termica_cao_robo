# Projeto: detecção de pessoas em térmica pura (SAR)

Iniciação Científica. Câmera Thermal Master P1 (160×120, radiométrica, **sem RGB**)
acoplada a um cão robô de busca e salvamento. O detector funde a forma (YOLO12s
fine-tunado) com a assinatura térmica (radiometria em °C/pixel) numa atualização em
razão de chances — ver o cabeçalho de `detect.py`.

## Leia isto primeiro

**`ROADMAP.md` é a fonte de verdade sobre o que fazer a seguir.** Ele tem o estado de
cada um dos 7 passos, as dependências entre eles, o critério objetivo de conclusão de
cada um, e um diário de sessões.

Ao começar: leia a tabela `#Estado atual` e pegue o primeiro passo `PENDENTE` cujos
pré-requisitos estejam `OK`.
Ao terminar: marque as tarefas, atualize o status e **escreva no `#Diário de sessões`**
— é o único mecanismo de continuidade entre sessões.

## Mapa dos arquivos

| Arquivo | Papel |
|---|---|
| `../../mapa_termo_semantico/` | **Repositório separado** (GitHub privado `ThiagoHSAl/mapa_termo_semantico`): Pioneer 3-DX + Hokuyo em ROS 2, roadmap `ROADMAP_PIONEER_SLAM.md` e código `pioneer/`. Movido para lá em 2026-09-24. |
| `detect.py` | Detector em operação. Fusão forma + térmica. Todos os limiares no topo. |
| `termica_comum.py` | Geometria do sensor, termometria, curva de spot-size, fusão. Compartilhado entre detecção e calibração — **se divergir, a calibração deixa de valer para o detector.** |
| `calibracao/` | `calibracao_temperatura.py` (coleta, Passo 1 — nunca executado) e `analisar_calibracao.py` (ajusta a curva e imprime os parâmetros para o `detect.py`). Dados em `calibracao/dados/`. |
| `EXPLICACAO_SISTEMA_TERMICO.txt` | Explicação geral do sistema, para leigos. |
| `ensaio_bancada/` | Ensaio de temperatura ao longo do tempo, temperatura × distância e gráficos. |
| `treinamento/train.py` | Receita do fine-tune. Dataset em `../datasets/ThermalCachorro/` (fora do git; 165 train / 70 valid, split com vazamento — ver Passo 2). |
| `thermal_person_finetune_rostos/` | Modelo **em produção** (usado pelo `detect.py`). |

Dependência externa: `p3_camera` em `~/p3-ir-camera/p3_camera.py`.

## Convenções

- Código, comentários e documentação em **português**.
- Comentários registram a **justificativa física e a alternativa rejeitada**, não o
  que a linha faz. Manter esse padrão — é o que já está praticamente escrito do artigo.
- Constante nova sem procedência medida entra marcada como provisória, com a razão.
- Nenhum passo do roadmap é dado por concluído sem o critério de conclusão satisfeito:
  sempre um número, um arquivo ou uma tabela — nunca "parece estar funcionando".
