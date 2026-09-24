import os
import csv
from datetime import datetime
import matplotlib

# Força o uso do protocolo de compatibilidade para evitar o warning do Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb" 
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt

# Procura o arquivo tanto na subpasta quanto na raiz
ARQUIVO_CSV = "ensaio_bancada/log_temperaturas.csv"
if not os.path.exists(ARQUIVO_CSV):
    ARQUIVO_CSV = "log_temperaturas.csv"

# Lista para armazenar apenas os dados desejados
tempos_minutos = []
temperaturas_centro = []

tempo_inicial = None

print(f"Lendo dados de {ARQUIVO_CSV}...")

try:
    with open(ARQUIVO_CSV, mode='r') as arquivo:
        leitor = csv.reader(arquivo, delimiter=',')
        cabecalho = next(leitor) # Pula a primeira linha
        
        for linha in leitor:
            if len(linha) < 7:
                continue
                
            # Extrai o momento da captura
            dt_captura = datetime.strptime(linha[2], "%Y%m%d_%H%M%S")
            
            if tempo_inicial is None:
                tempo_inicial = dt_captura
            
            # Tempo em minutos
            diferenca_segundos = (dt_captura - tempo_inicial).total_seconds()
            tempo_minutos = diferenca_segundos / 60.0
            
            # Costura apenas a temperatura do centro
            temp_centro = float(linha[3] + "." + linha[4])
            
            tempos_minutos.append(tempo_minutos)
            temperaturas_centro.append(temp_centro)

    print(f"Leitura concluída. {len(tempos_minutos)} pontos plotados.")

    # ==========================================
    # GERAÇÃO DO GRÁFICO
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Plota apenas a linha do Centro
    plt.plot(tempos_minutos, temperaturas_centro, label="Temperatura no Centro (°C)", 
             color="green", linewidth=2.5, linestyle='-')

    # Configuração dos títulos e eixos
    plt.title("Estabilidade Térmica da Câmera P1 - Ensaio de Bancada", fontsize=14, fontweight='bold')
    plt.xlabel("Tempo de Operação (minutos)", fontsize=12)
    plt.ylabel("Temperatura (°C)", fontsize=12)

    # Travando os limites do gráfico
    max_tempo = int(max(tempos_minutos)) if tempos_minutos else 60
    plt.xlim(left=0, right=max_tempo)
    plt.ylim(0, 50) # LIMITANDO O EIXO Y DE 0 A 100 GRAUS

    # --- NOVAS LINHAS DE CONTROLE DE ESCALA ---
    # Força o Eixo X a pular de 2 em 2 minutos
    plt.xticks(range(0, max_tempo + 2, 2))
    
    # Força o Eixo Y a pular de 1 em 1 grau (indo do 0 ao 100)
    plt.yticks(range(0, 51, 1))

    # Grade
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Legenda
    plt.legend(loc="upper right", fontsize=11)

    # Exibição
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Erro: O arquivo '{ARQUIVO_CSV}' não foi encontrado.")
except Exception as e:
    print(f"Erro inesperado: {e}")
