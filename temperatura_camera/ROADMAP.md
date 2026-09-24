# ROADMAP — Detecção de pessoas em térmica pura (SAR / Thermal Master P1)

Plano de execução em 7 passos, derivado do confronto entre este projeto e 5 artigos
da literatura atual (ver `#Referências` no final). Foi escrito para ser executado ao
longo de **várias sessões independentes**, por pessoas ou sessões de agente que não
acompanharam as anteriores.

---

## COMO USAR ESTE ARQUIVO

**Ao começar uma sessão:**

1. Leia a tabela `#Estado atual`. O próximo passo é o primeiro com status `PENDENTE`
   cujos pré-requisitos estejam todos `OK`.
2. Vá até a seção daquele passo. Ela é auto-contida: tem o objetivo, a justificativa
   na literatura, os comandos, os arquivos envolvidos e o critério objetivo de
   conclusão. Não é preciso reler os artigos nem as sessões anteriores.
3. Execute só aquele passo. Passos fora de ordem quebram as dependências.

**Ao terminar uma sessão (mesmo sem terminar o passo):**

1. Marque as tarefas concluídas (`- [ ]` → `- [x]`).
2. Atualize o status do passo na tabela `#Estado atual`.
3. Acrescente uma entrada em `#Diário de sessões` — o que foi feito, o número que
   saiu, e o que travou. **Este é o único mecanismo de continuidade entre sessões.**
   Uma sessão que não escreve no diário faz a próxima repetir trabalho.

**Regra de ouro:** nenhum passo é dado por concluído sem o **critério de conclusão**
satisfeito. O critério é sempre um número, um arquivo ou uma tabela — nunca "parece
estar funcionando".

---

## Estado atual

| # | Passo | Status | Pré-requisitos | Custo |
|---|---|---|---|---|
| 0 | Análise da literatura e diagnóstico | **OK** (2026-08-28) | — | — |
| 1 | Rodar a calibração térmica (coleta + ajuste) | PENDENTE | 0 | ~3 dias de coleta |
| 2 | Re-split por sessão + test set retido + re-medição | PENDENTE | 0 | ~4 h |
| 3 | Tabela de ablação YOLO-only / térmica-only / fundido | PENDENTE | 1, 2 | ~2 dias |
| 4 | Rescaling próprio a partir do `celsius` (Fieldscale) | PENDENTE | 3 | ~1 semana |
| 5 | Consistência temporal multi-frame | PENDENTE | 3 | ~4 h |
| 6 | `t_ambiente` como campo 2D + evidência por gradiente | PENDENTE | 4 | ~1 semana |
| 7 | Emissividade e temperatura refletida | PENDENTE | 1 | ~3 h |

Status possíveis: `PENDENTE` · `EM ANDAMENTO` · `OK` · `BLOQUEADO (motivo)`

### Dependências

```
        0 (feito)
        │
   ┌────┴────┐
   1         2          1 e 2 são independentes — podem ser feitos em
   │         │          qualquer ordem, ou em paralelo.
   ├──> 7    │
   │         │
   └────┬────┘
        3   <-- CONGELA O BASELINE. Nada abaixo faz sentido sem esta tabela.
        │
   ┌────┴────┐
   4         5
   │
   6
```

**Por que o 3 é o pivô:** os passos 4, 5 e 6 mudam a entrada ou a decisão do detector.
Sem a tabela do passo 3 medida **antes**, não há como demonstrar que qualquer um deles
melhorou alguma coisa. O passo 3 não é burocracia — é o instrumento de medida de todo
o resto, e é a tabela que vai para o artigo.

---

## Passo 0 — Análise da literatura e diagnóstico ✔ CONCLUÍDO

Feito em 2026-08-28. Conclusões que os passos seguintes atacam:

**O que já está alinhado com a literatura** (preservar, não mexer):
- Fusão *decision-level* de dois ramos independentes (arquitetura do DualFOD) — e a
  fusão em razão de chances é **superior** à do DualFOD, que faz união pura e por isso
  só consegue acrescentar detecção, nunca derrubar.
- Decidir sobre a radiometria absoluta (°C/pixel) e não sobre a imagem colorizada — é
  a tese central do FLAME 3 (TIFF 91.38% > térmica JPEG 84.05%).
- Ramo térmico não-supervisionado com filtros geométricos + validação por contraste.
- ΔT relativo ao fundo mascarado, em vez de limiar absoluto fixo.
- Modelagem explícita do spot-size (nenhum dos 5 artigos faz isso; só reportam).
- `hsv_h=0, hsv_s=0, hsv_v=0.4` no treino — coerente com o artigo SPIE.

**Os 7 problemas** viram os 7 passos abaixo.

---

## Passo 1 — Rodar a calibração térmica

> **Status:** PENDENTE · **Pré-req:** nenhum · **Custo:** ~3 dias de coleta

### Objetivo
Substituir por medida os parâmetros do `detect.py` que hoje são chute declarado:
`DT_VIVO_PERTO`, `DT_VIVO_LONGE`, `EXPOENTE_LIMIAR`, `P_TERMICA_MIN`, `P_TERMICA_MAX`.

### Por quê
`calibracao_termica/` **não existe** — zero amostras coletadas. Toda a régua de decisão
do sistema repousa em números inventados, e não existe nenhuma taxa de falso positivo
para apresentar. O DualFOD faz o mínimo exigível: varre ΔT ∈ {2,3,5,7,10} °C e tabula
TP/FP/miss (ΔT=2 → precisão 0.55; ΔT=5 → 0.89; ΔT=7 → 1.00 mas perde 6 de 12).
Meia página de artigo — e é o que sustenta a escolha do limiar.

**A infraestrutura já está escrita e é boa.** Ela só nunca foi executada.

### Tarefas

- [ ] Sessão de coleta em cômodo **FRIO** (~20 °C), classe `viva`:
      `python3 calibracao_temperatura.py --local sala_fria`
      Distâncias 1, 2, 3, 4, 5, 6, 8, 10, 12 m · ~10 amostras por distância ·
      variando pose (`p`) e vestuário (`v`) · `s` para NUC **antes de cada bloco**.
- [ ] Mesma coleta em cômodo **QUENTE** (>36 °C) — define o regime `quente`.
- [ ] Coleta na faixa de **CROSSOVER** (33–37 °C), se conseguir produzi-la.
- [ ] **NEGATIVOS, em quantidade comparável aos positivos** (`k` alterna a classe):
      `negativo_manequim` (manequim, casaco pendurado, mochila),
      `negativo_fonte` (radiador, notebook, cano, luminária),
      `negativo_reflexo` (pessoa em vidro/espelho/inox),
      `negativo_outro`.
- [ ] Proxies de triagem: `proxy_resfriando` (recipiente a ~35 °C, `r` zera o
      cronômetro no início) e `proxy_ambiente`.
- [ ] `python3 analisar_calibracao.py`
- [ ] Colar os parâmetros impressos no topo do `detect.py`, **trocando os comentários
      de "PROVISÓRIOS" por a data e o n da coleta que os produziu**.

### Critério de conclusão
`calibracao_termica/amostras.csv` existe, com ≥ 300 amostras `viva` cobrindo os 9
pontos de distância em ≥ 2 regimes, e ≥ 200 amostras `negativo_*`.
`analisar_calibracao.py` roda sem avisos de amostra insuficiente e os 5 parâmetros
estão no `detect.py` com procedência anotada.

### Armadilhas
- **Sem negativos não existe `P_TERMICA_MIN`.** A estimativa de plausibilidade supõe
  as duas classes igualmente representadas. Coletar só positivos desperdiça a coleta.
- **Os pontos distantes (8, 10, 12 m) são os que mais importam** e os mais chatos de
  coletar. Sem eles `DT_VIVO_LONGE` sai otimista e o detector perde vítima longe.
- **NUC antes de cada bloco.** A deriva entre NUCs entra na amostra disfarçada de
  efeito de distância, e a calibração passa a medir as duas coisas somadas.
- Cada amostra grava o `.npy` bruto. **Não apague** — é o que permite reanalisar
  (inclusive aplicar o Passo 7 retroativamente, sem recoletar).

### Arquivos
`calibracao_temperatura.py` · `analisar_calibracao.py` · `termica_comum.py` ·
saída em `calibracao_termica/`

---

## Passo 2 — Re-split por sessão, test set retido, re-medição

> **Status:** PENDENTE · **Pré-req:** nenhum · **Custo:** ~4 h

### Objetivo
Descobrir o mAP verdadeiro do detector. Provavelmente vai doer.

### Por quê
Evidência medida em 2026-08-28: o dataset `ThermalCachorro/` tem **3 datas de
captura** e **todas as três aparecem em train e em valid**:

| data | train | valid |
|---|---|---|
| 260624 | 105 | 48 |
| 260709 | 27 | 8 |
| 260710 | 33 | 14 |

O split é aleatório por frame, e os frames são consecutivos (`_140152`, `_140153`,
`_140154`…). Quadros quase idênticos caem dos dois lados. `test/images` está **vazio**.

O sintoma está no próprio disco: o modelo base (`yolo12s_thermal-3`, dataset grande)
fez **mAP50 = 0.436**; o fine-tune de 165 imagens fez **0.905**. Ganho de +47 pontos
com 165 imagens não é aprendizado, é vazamento.

Referência: o DualFOD tem test set retido **e** validação cross-site em outro
aeródromo. O FLAME 3 treina em queimadas e testa em queimadas diferentes.

### Tarefas

- [ ] **Localizar o dataset do modelo em produção.** O `detect.py` usa
      `thermal_person_finetune_rostos/weights/best.pt`, treinado em `/workspace/`
      (container) sobre um dataset "rostos" que **não está neste disco** — só o
      `ThermalCachorro/` (165/70) está. Sem achá-lo, o passo roda sobre o
      `ThermalCachorro` e o modelo em produção fica sem medida honesta.
      → Se não for encontrado, marcar o passo `BLOQUEADO` e registrar no diário.
- [ ] Re-split **por sessão de captura**, nunca aleatório por frame:
      treino = 260624 · validação = 260709 · **teste = 260710** (retido, tocado uma
      vez só no final). Ajustar conforme o volume real de cada sessão.
- [ ] Coletar ≥ 1 sessão nova, em **outro cômodo e com outra pessoa**, exclusiva para
      teste. É o equivalente à *cross-site validation* do DualFOD e vale mais que
      qualquer ganho de mAP no split interno.
- [ ] Retreinar com `ThermalCachorro/train.py` (hiperparâmetros já estão adequados).
- [ ] Medir no test set retido: `yolo val model=<best.pt> data=<data.yaml> split=test`
- [ ] Registrar no diário **os dois números**: mAP50 antigo (0.905, inflado) e o novo.

### Critério de conclusão
Existe um `data.yaml` com `train`/`val`/`test` disjuntos **por sessão**, e o diário
registra o mAP50 / mAP50-95 medidos no split de teste retido, tocado uma única vez.

### Armadilha
Não ajuste hiperparâmetro olhando o test set. Se mexer, o test set virou validação e
o número perdeu o valor.

### Arquivos
`ThermalCachorro/data.yaml` · `ThermalCachorro/train.py` · `thermal_person_finetune_rostos/`

---

## Passo 3 — Tabela de ablação: YOLO-only / térmica-only / fundido

> **Status:** PENDENTE · **Pré-req:** 1, 2 · **Custo:** ~2 dias

### Objetivo
Produzir o número que sustenta a tese central do projeto — e que hoje não existe.

### Por quê
O trabalho afirma que a térmica reforça a decisão em três direções (derruba FP,
resgata FN, propõe o que o YOLO não viu). **Nenhuma medida sustenta isso.** Os
contadores do HUD (`resgates`, `derrubados`, `cand.T`) estão lá, mas sem ground truth
e sem protocolo são só números na tela.

É a Tabela 10 do DualFOD, adaptada. **Esta tabela é o artigo.**

### Tabela a produzir

| Distância | Regime | GT | YOLO-only | Térmica-only | Fundido | Resgates | Derrubadas corretas | FP |
|---|---|---|---|---|---|---|---|---|
| 2 m | frio | | | | | | | |
| 5 m | frio | | | | | | | |
| … | | | | | | | | |

Varrer: distância (2–12 m) × regime (frio / crossover / quente) × as classes de
negativo do Passo 1.

### Tarefas

- [ ] Definir o protocolo de ground truth: cenas encenadas com posição e contagem de
      pessoas conhecidas, incluindo **poses de vítima** (deitada, encolhida, de costas,
      parcialmente coberta) e negativos plantados no quadro.
- [ ] Gravar as cenas em `.npy` bruto + vídeo, para reprocessar sem recoletar.
- [ ] Escrever `avaliar_fusao.py`: roda os três modos sobre as mesmas cenas —
      (a) YOLO puro em `LIMIAR_ACEITE`, (b) só o ramo térmico
      (`propor_anomalias_termicas` sem YOLO), (c) o pipeline fundido — e emite a tabela.
- [ ] **Análise de sensibilidade** ao estilo DualFOD: varrer `LIMIAR_ACEITE` e
      `P_TERMICA_MAX` e tabular TP/FP/miss em cada ponto.
- [ ] Congelar o resultado como **BASELINE** e registrá-lo no diário. Todo passo
      seguinte se compara a ele.

### Critério de conclusão
`avaliar_fusao.py` existe, a tabela está preenchida com ground truth real, e o diário
registra a linha de baseline. A partir daqui, qualquer alteração no detector é medida
contra esses números.

### Arquivos
novo: `avaliar_fusao.py` · consome `detect.py`, `termica_comum.py`

---

## Passo 4 — Rescaling próprio a partir do `celsius` (Fieldscale)

> **Status:** PENDENTE · **Pré-req:** 3 · **Custo:** ~1 semana

### Objetivo
Parar de alimentar o YOLO com a AGC proprietária do hardware.

### Por quê
```python
frame = cv2.cvtColor(ir_brightness, cv2.COLOR_GRAY2BGR)   # 8-bit AGC do hardware
```
Você tem `thermal` (uint16) e `celsius` em mãos e usa a radiometria **só na fusão**.
O detector — o ramo que carrega a evidência de forma — vê a saída de uma LUT 1D global
proprietária: exatamente a categoria que o Fieldscale mede e supera (incluindo a AGC
da FLIR, em todos os limiares de IoU e faixas de tamanho; ablação 45.5 → 46.5 AP).

Pior no caso SAR: **AGC depende da cena inteira**. Entra um foco de incêndio no canto
do quadro e a mesma vítima muda de aparência entre frames consecutivos. Num prédio em
chamas isso não é hipótese, é o cenário de uso.

Maior relação ganho/esforço do roadmap, e é o passo publicável: *"Fieldscale aplicado
a SAR indoor com sensor de 160×120"*. Código dos autores é aberto:
https://github.com/hyeonjaegil/fieldscale

### Tarefas

- [ ] Implementar em `termica_comum.py` a função `rescalar(celsius) -> uint8`.
      Comece pela versão simples: janela fixa ancorada na banda humana + CLAHE.
      Já é auditável e determinística, ao contrário da AGC.
- [ ] Implementar o Fieldscale propriamente: pooling min/max em grid (N=7 para cena
      ampla, 4×4 para interior estreito) → supressão de extremos locais (T_LES≈100) →
      message passing → interpolação bilinear → γ=1.5 → CLAHE.
- [ ] **Retreinar o detector** sobre imagens rescaladas pelo método novo. O modelo
      atual aprendeu a textura da AGC; trocar a entrada sem retreinar mede a coisa
      errada.
- [ ] Re-rodar a tabela do Passo 3 e comparar com o baseline congelado.
- [ ] Medir o tempo por frame — o Fieldscale roda a 72–323 Hz em i7; confirmar no
      hardware do robô.

### Critério de conclusão
Tabela do Passo 3 re-medida com o rescaling novo, lado a lado com o baseline, e o
tempo por frame compatível com a operação do robô.

### NÃO faça
**Não use SAHI.** Ela resolve "objeto pequeno em imagem grande" (DualFOD, 5–30 m de
altitude, imagens de dezenas de MP). Sua imagem tem 160×120 — fatiar é
contraproducente.

### Arquivos
`termica_comum.py` (função nova) · `detect.py` (troca da linha de entrada) ·
`calibracao_temperatura.py` (mesma entrada, para não divergir) · `ThermalCachorro/train.py`

---

## Passo 5 — Consistência temporal multi-frame

> **Status:** PENDENTE · **Pré-req:** 3 · **Custo:** ~4 h

### Objetivo
Exigir persistência antes de aceitar uma proposta térmica.

### Por quê
Tudo hoje é frame a frame. O DualFOD recomenda explicitamente, como trabalho futuro,
*"temporal filtering and multi-frame consistency checks to suppress transient anomalies
in thermal frames"*; o Fieldscale reporta flicker temporal como sua principal limitação.

No seu caso é mais forte que nos dois, porque **o robô se move**: um blob que persiste
por N frames com deslocamento coerente vale ordens de magnitude mais que um blob
isolado. E a deriva entre NUCs — que você já modelou em `RelogioNuc` — é exatamente um
artefato transitório que a filtragem temporal remove de graça.

`MAX_PROPOSTAS = 5` por frame é um remendo no lugar disso.

### Tarefas

- [ ] Rastreador leve de propostas entre frames (associação por IoU/centroide; não
      precisa de Kalman para começar).
- [ ] Exigir persistência ≥ N frames (começar com N=3) antes de exibir `[T] CANDIDATO`.
- [ ] Anotar no HUD há quantos frames a proposta persiste.
- [ ] Suprimir propostas nascidas logo após um NUC (o shutter perturba a leitura).
- [ ] Re-rodar a tabela do Passo 3: a métrica que deve cair é **FP**, sem queda em
      resgates. Se resgates caírem, N está alto demais.

### Critério de conclusão
FP do ramo térmico reduzido contra o baseline, com resgates preservados, medido na
tabela do Passo 3.

### Arquivos
`detect.py` (loop) · `termica_comum.py` (classe de rastreamento)

---

## Passo 6 — `t_ambiente` como campo 2D + evidência por gradiente

> **Status:** PENDENTE · **Pré-req:** 4 · **Custo:** ~1 semana

### Objetivo
Eliminar as duas simplificações que mais custam informação: um ambiente escalar para
o frame inteiro, e um percentil único como evidência térmica de uma caixa.

### Por quê

**(a) O ambiente escalar é o erro do Fieldscale, um nível acima.**
```python
return float(np.median(fundo))   # UM número para o frame inteiro
```
Mascarar os candidatos antes da mediana foi um acerto real. Mas o resultado ainda é um
escalar global, e dele sai o `regime` do cômodo inteiro. Num prédio colapsado com foco
de incêndio num canto e corredor frio no outro, um único "ambiente" é falso — é a mesma
falha conceitual que o Fieldscale ataca na LUT 1D, aplicada à estimativa de fundo.

**(b) Um percentil só joga fora a estrutura espacial.**
```python
t_alvo = np.percentile(regiao, 99)   # um número para a caixa inteira
```
Um corpo humano não é isotérmico: cabeça e mãos expostas ficam bem acima do tronco
vestido. Esse **gradiente interno sobrevive ao crossover** — quando o contraste médio
contra o fundo desaparece, a assinatura interna não. Hoje o crossover devolve
`P_TERMICA_NEUTRA` e `propor_anomalias_termicas` retorna `[]`: você trata "não há
informação **nesta estatística**" como "não há informação". E o crossover é justamente
onde a SAR mais precisa.

### Tarefas

- [ ] `temperatura_ambiente` devolve um **campo 2D** (mesma máquina do Passo 4: pooling
      em grid + message passing), não um escalar. Manter a assinatura antiga como
      wrapper para não quebrar `calibracao_temperatura.py`.
- [ ] ΔT e `regime` avaliados **localmente por região**, não para o frame todo.
- [ ] Evidência térmica por **gradiente/histograma** da região, além do percentil.
- [ ] Reabilitar a térmica no crossover usando a assinatura interna.
- [ ] Re-rodar a tabela do Passo 3, com atenção às linhas de **regime crossover** — é
      onde o ganho deve aparecer.

### Critério de conclusão
As linhas de crossover da tabela do Passo 3 deixam de ser degeneradas (térmica-only > 0
e resgates > 0), sem aumento de FP nos outros regimes.

### Arquivos
`termica_comum.py` (`temperatura_ambiente`, `plausibilidade_*`) · `detect.py`

---

## Passo 7 — Emissividade e temperatura refletida

> **Status:** PENDENTE · **Pré-req:** 1 · **Custo:** ~3 h

### Objetivo
Fechar o buraco físico da banda absoluta.

### Por quê
```python
celsius = raw_to_celsius(thermal)   # (raw/64) - 273.15, sem correção
```
Mas `T_HUMANO_MIN/MAX = 28–40 °C` é uma afirmação sobre **temperatura absoluta**.
Pele tem ε ≈ 0.98; algodão ~0.77–0.95; alumínio e vidro de escombro, muito menos — e
refletem. A `p3_camera` **já expõe** `raw_to_celsius_corrected(raw, EnvParams)` com
emissividade, temperatura refletida, distância e umidade, e ela não é usada.

O DualFOD lista *"irregular surface emissivity, sunlight reflections"* como a
**principal** fonte de falso positivo do ramo térmico dele.

### Tarefas

- [ ] Reprocessar os `.npy` do Passo 1 com `raw_to_celsius_corrected`, varrendo
      ε ∈ {0.95, 0.98} e temperatura refletida. **Não precisa recoletar** — é por isso
      que o Passo 1 guarda os frames brutos.
- [ ] Quantificar o deslocamento em °C que a correção provoca na banda humana.
- [ ] Decidir com o número na mão: (a) adotar a correção no `detect.py` com
      `EnvParams` fixo, ou (b) manter a leitura simples e documentar o erro como
      incerteza sistemática **já embutida na largura da banda**.
- [ ] Registrar a decisão e o número no `EXPLICACAO_SISTEMA_TERMICO.txt`.

### Critério de conclusão
O comentário da banda `T_HUMANO_MIN/MAX` no `detect.py` cita um erro de emissividade
**medido**, não estimado — qualquer que seja a decisão tomada.

### Arquivos
`~/p3-ir-camera/p3_camera.py` (`EnvParams`, `raw_to_celsius_corrected`) ·
`detect.py` · `termica_comum.py`

---

## Fora do escopo (decisões já tomadas — não reabrir sem motivo novo)

- **Térmica pura, sem RGB.** A Tabela VII do SuperYOLO mostra que IR-only perde de
  RGB-only nos 8 detectores testados, por 7–10 pontos de mAP50 (SuperYOLO: IR 65.60 /
  RGB 72.49 / Multi 75.09). A escolha aqui é justificada pelo caso de uso — escuro,
  fumaça, poeira eliminam o RGB — mas **isso precisa estar escrito no artigo com essas
  referências citadas**, senão o primeiro revisor pergunta e não há resposta pronta.
  → Se o cão robô ganhar qualquer câmera RGB ou NIR, vale um terceiro voto: a fusão em
  odds já aceita, é só mais um fator multiplicativo, e o DualFOD dá o algoritmo de
  associação espacial pronto (Algoritmo 1).
- **SAHI:** inaplicável a 160×120. Ver Passo 4.
- **Branch de super-resolução (SuperYOLO):** ideia boa (+2.2 mAP50 no YOLOv5s, custo
  zero na inferência porque o branch é descartado), mas depende de todo o resto estar
  medido. Reavaliar depois do Passo 6.

---

## Diário de sessões

<!-- Formato: ## AAAA-MM-DD — Passo N — quem
     O que foi feito · o número que saiu · o que travou · o que fazer a seguir.
     Append-only: nunca edite entradas antigas, só acrescente. -->

## 2026-08-28 — Passo 0 — Thiago + Claude
Análise das 5 referências de `~/Referencias_radiometria` confrontada com `detect.py`,
`termica_comum.py`, `calibracao_temperatura.py` e os três runs de treino.
Diagnóstico: 7 problemas, virados nos 7 passos acima. Este arquivo criado.

Fatos medidos nesta sessão (não repetir a verificação):
- `calibracao_termica/` não existe — zero amostras coletadas.
- `ThermalCachorro/`: 3 datas de captura, **todas presentes em train e valid**
  (260624: 105/48 · 260709: 27/8 · 260710: 33/14). `test/images` vazio.
- `yolo12s_thermal-3` (base): mAP50 0.436 · `thermal_person_finetune`: 0.892 ·
  `thermal_person_finetune_rostos` (em produção): 0.905 — os dois últimos inflados.
- O dataset "rostos" do modelo em produção **não está neste disco** (treino rodou em
  `/workspace/`). É pré-requisito do Passo 2.
- `p3_camera.py` tem `raw_to_celsius_corrected` + `EnvParams`; `detect.py` não usa.

Próximo: Passo 1 ou Passo 2 (independentes).

---

## Referências

Em `~/Referencias_radiometria/`:

1. **DualFOD** — Ahmed, Caldwell, Khalid. *Drones* 10(3):225, 2026.
   YOLO12+SAHI no RGB · ramo térmico não-supervisionado (black-hat + DoG + MSER,
   consenso 2/3) · fusão decision-level por proximidade (τ=50 px) · ΔT>5 °C.
   → Passos 3, 5, 7
2. **SuperYOLO** — Zhang et al. *IEEE TGRS*, 2023.
   Fusão pixel-level RGB+IR · branch de super-resolução descartado na inferência ·
   Tabela VII (IR-only vs RGB-only vs Multi).
   → Fora do escopo; reavaliar após Passo 6
3. **FLAME 3** — Hopkins et al., arXiv:2412.02831, 2024.
   TIFF radiométrico vs JPEG colorizado (91.38% vs 84.05%) · rotulagem por limiar de
   temperatura.
   → Justifica a arquitetura atual; Passos 4 e 7
4. **Fieldscale** — Gil, Jeon, Kim. arXiv:2405.15395, 2024.
   Rescaling 14→8 bits com campos 2D · https://github.com/hyeonjaegil/fieldscale
   → Passos 4 e 6
5. **Infrared object detection … YOLOv8 with data augmentation** — Zhao. *SPIE* 13792, 2025.
   FLIR_ADAS_v2 · augment fotométrico pré-computado · a classe definida por
   intensidade piorou 2,3%.
   → Já incorporado no treino atual
