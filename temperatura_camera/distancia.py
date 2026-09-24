import os
import matplotlib

# Força o uso do protocolo de compatibilidade para evitar o warning do Wayland no Ubuntu
os.environ["QT_QPA_PLATFORM"] = "xcb" 
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. DADOS HARDCODED (Coletados no corredor)
# ==========================================
distancias = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
temperaturas = [26.1, 25.6, 25.4, 25.2, 25.1, 25.0, 24.9, 25.1, 25.2, 24.6, 24.8, 24.9, 24.8, 24.7, 25.1, 25.2, 25.5, 25.5, 25.3, 25.1, 24.8]

# ==========================================
# 2. GERAÇÃO DO GRÁFICO
# ==========================================
plt.figure(figsize=(10, 6))

# Plota a linha com "bolinhas" marcando o ponto exato da medição
plt.plot(distancias, temperaturas, marker='o', color='blue', linewidth=2, markersize=6, label="Leitura da Parede (Alvo Infinito)")

# Configuração de Título e Eixos
plt.title("Avaliação de Leitura Radiométrica por Distância", fontsize=14, fontweight='bold')
plt.xlabel("Distância da Câmera (metros)", fontsize=12)
plt.ylabel("Temperatura Medida (°C)", fontsize=12)

# --- ESCALA DO EIXO X ---
# Força o eixo X ir de 0 a 42, pulando de 2 em 2 metros
plt.xticks(range(0, 42, 2))

# --- ESCALA DO EIXO Y ---
# "Zoom" no eixo Y para enxergar o ruído (vai de 24.0 a 26.5)
plt.yticks([i/10 for i in range(150, 400, 10)]) # Pula de 0.5 em 0.5 grau
plt.ylim(15.0, 40.0)

# Adiciona uma linha vermelha tracejada mostrando o valor "estabilizado" (média a partir de 2m)
# Ignoramos o ponto de 1m, pois provavelmente sofreu reflexo do corpo do operador
media_estavel = sum(temperaturas[1:]) / len(temperaturas[1:])
plt.axhline(y=media_estavel, color='red', linestyle=':', label=f"Média Estável ({media_estavel:.2f}°C)")

# Grade de fundo para facilitar a leitura
plt.grid(True, linestyle='--', alpha=0.7)

# Legenda
plt.legend(loc="upper right", fontsize=11)

# Exibição
plt.tight_layout()
plt.show()
