import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime, timedelta
from p3_camera import Model, P3Camera, get_model_config, raw_to_celsius

# --- CONFIGURAÇÕES DO ENSAIO ---
INTERVALO = 5  # segundos
DURACAO_TOTAL = 3600  # 1 hora em segundos
MODELO = Model.P1
PASTA_DESTINO = os.path.dirname(os.path.abspath(__file__))  # grava ao lado deste script
ARQUIVO_CSV = os.path.join(PASTA_DESTINO, "log_temperaturas.csv")

if not os.path.exists(PASTA_DESTINO):
    os.makedirs(PASTA_DESTINO)

# Inicialização da Câmera
camera = P3Camera(config=get_model_config(MODELO))
camera.connect()
camera.init()
camera.start_streaming()

cv2.namedWindow("Monitoramento de Bancada", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Monitoramento de Bancada", 800, 600)

tempo_inicio = datetime.now()
tempo_final = tempo_inicio + timedelta(seconds=DURACAO_TOTAL)
total_fotos = int(DURACAO_TOTAL / INTERVALO)

print(f"Iniciando ensaio de bancada: {total_fotos} leituras planejadas.")
print(f"Salvando log em: {ARQUIVO_CSV}")

# Cria e prepara o arquivo CSV com os cabeçalhos
with open(ARQUIVO_CSV, mode='w', newline='') as arquivo_log:
    escritor_csv = csv.writer(arquivo_log, delimiter=';')
    escritor_csv.writerow(['Data', 'Hora', 'Timestamp_Sistema', 'Temp_Centro_C', 'Temp_Max_C'])

contagem = 0

try:
    while datetime.now() < tempo_final:
        proxima_captura = time.time() + INTERVALO
        
        ir_brightness, thermal_raw = camera.read_frame_both()
        
        if thermal_raw is not None and ir_brightness is not None:
            contagem += 1
            agora = datetime.now()
            data_str = agora.strftime("%Y-%m-%d")
            hora_str = agora.strftime("%H:%M:%S")
            timestamp = agora.strftime("%Y%m%d_%H%M%S")
            
            # Conversão para Celsius
            temp_celsius = raw_to_celsius(thermal_raw)
            temp_max = np.max(temp_celsius)
            
            # Pega a temperatura exatamente no centro (onde presumivelmente está o termopar/alvo)
            h, w = temp_celsius.shape
            temp_centro = temp_celsius[h // 2, w // 2]
            
            # Salva a linha no CSV
            with open(ARQUIVO_CSV, mode='a', newline='') as arquivo_log:
                escritor_csv = csv.writer(arquivo_log, delimiter=';')
                # Troca ponto por vírgula se for abrir no Excel em PT-BR
                escritor_csv.writerow([data_str, hora_str, timestamp, f"{temp_centro:.2f}".replace('.', ','), f"{temp_max:.2f}".replace('.', ',')])
            
            # Salva os backups brutos
            np.save(os.path.join(PASTA_DESTINO, f"raw_{timestamp}.npy"), thermal_raw)
            cv2.imwrite(os.path.join(PASTA_DESTINO, f"vis_{timestamp}.jpg"), ir_brightness)
            
            print(f"[{contagem}/{total_fotos}] {hora_str} | Centro: {temp_centro:.1f}°C | Max: {temp_max:.1f}°C")

            # --- VISUALIZAÇÃO ---
            img_suave = cv2.resize(ir_brightness, (640, 480), interpolation=cv2.INTER_CUBIC)
            img_colorida = cv2.applyColorMap(img_suave, cv2.COLORMAP_INFERNO)
            
            # Mira indicando o pixel central exato
            centro_tela = (320, 240)
            cv2.drawMarker(img_colorida, centro_tela, (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            
            cv2.putText(img_colorida, f"Leitura: {contagem}/{total_fotos}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img_colorida, f"Centro: {temp_centro:.1f} C", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
            cv2.imshow("Monitoramento de Bancada", img_colorida)

        # Controle de tempo
        wait_time = proxima_captura - time.time()
        if wait_time > 0:
            if cv2.waitKey(int(wait_time * 1000)) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(1)

except Exception as e:
    print(f"\nErro no ensaio: {e}")

finally:
    print(f"\nEnsaio finalizado. Dados exportados para '{ARQUIVO_CSV}'.")
    camera.stop_streaming()
    camera.disconnect()
    cv2.destroyAllWindows()
