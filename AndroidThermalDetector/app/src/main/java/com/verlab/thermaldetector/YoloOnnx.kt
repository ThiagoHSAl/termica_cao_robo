package com.verlab.thermaldetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** Uma detecção em coordenadas da TELA (pixels). */
data class Detection(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float
)

/**
 * Carrega o modelo YOLO (best.onnx) e roda inferência sobre um Bitmap da tela.
 *
 * Modelo: entrada [1,3,640,640] RGB normalizado 0-1; saída [1,5,8400]
 * (cx, cy, w, h em pixels de 640 + score de "person"). NMS feito aqui.
 */
class YoloOnnx(context: Context) {

    companion object {
        // Aceleração por hardware (NNAPI). true = tenta NPU/GPU; false = só CPU.
        // OBS: para YOLOv12 (tem camadas de atenção) o NNAPI costuma piorar,
        // pois quebra o modelo e fica copiando dados CPU<->acelerador.
        const val USE_NNAPI = false
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    private val inputSize = 320   // menor entrada -> ~4x mais rápido na CPU
    // Buffer reutilizado a cada frame (evita alocar 1*3*640*640 floats sempre).
    private val inputBuffer: FloatBuffer =
        FloatBuffer.allocate(1 * 3 * inputSize * inputSize)

    var confThreshold = 0.35f
    var iouThreshold = 0.45f

    // Tempo da última inferência (ms) — exibido no overlay.
    var lastInferenceMs: Long = 0
        private set

    init {
        val modelBytes = context.assets.open("best.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            // Tenta usar o acelerador do aparelho (NPU/GPU/DSP via NNAPI).
            // Se o modelo/hardware não suportar, cai de volta pra CPU sozinho.
            // Coloque USE_NNAPI = false se ficar MAIS lento no seu aparelho.
            if (USE_NNAPI) {
                try {
                    addNnapi()
                } catch (e: Throwable) {
                    e.printStackTrace()
                }
            }
        }
        session = env.createSession(modelBytes, opts)
        inputName = session.inputNames.iterator().next()
    }

    /**
     * Roda a detecção. Recebe o Bitmap da tela inteira e devolve as caixas
     * já em coordenadas da tela (mesmo tamanho do bitmap de entrada).
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val srcW = bitmap.width
        val srcH = bitmap.height

        // --- Letterbox: encaixa a tela em 640x640 mantendo proporção ---
        val scale = min(inputSize.toFloat() / srcW, inputSize.toFloat() / srcH)
        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()
        val padX = (inputSize - newW) / 2f
        val padY = (inputSize - newH) / 2f

        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val pixels = IntArray(newW * newH)
        resized.getPixels(pixels, 0, newW, 0, 0, newW, newH)

        // Preenche o buffer CHW (R, depois G, depois B), fundo cinza (114/255).
        val buf = inputBuffer
        buf.rewind()
        val gray = 114f / 255f
        val area = inputSize * inputSize
        val arr = buf.array()
        // fundo
        java.util.Arrays.fill(arr, 0, area, gray)                 // R
        java.util.Arrays.fill(arr, area, 2 * area, gray)          // G
        java.util.Arrays.fill(arr, 2 * area, 3 * area, gray)      // B
        val offX = padX.toInt()
        val offY = padY.toInt()
        for (y in 0 until newH) {
            val row = (offY + y) * inputSize + offX
            for (x in 0 until newW) {
                val p = pixels[y * newW + x]
                val idx = row + x
                arr[idx] = ((p shr 16) and 0xFF) / 255f              // R
                arr[area + idx] = ((p shr 8) and 0xFF) / 255f        // G
                arr[2 * area + idx] = (p and 0xFF) / 255f            // B
            }
        }
        resized.recycle()

        // --- Inferência ---
        val t0 = System.currentTimeMillis()
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val tensor = OnnxTensor.createTensor(env, buf, shape)
        val output = session.run(mapOf(inputName to tensor))
        lastInferenceMs = System.currentTimeMillis() - t0
        val out = (output[0].value as Array<Array<FloatArray>>)[0] // [5][8400]
        tensor.close()
        output.close()

        val numAnchors = out[0].size // 8400
        val candidates = ArrayList<Detection>()
        for (i in 0 until numAnchors) {
            val score = out[4][i]
            if (score < confThreshold) continue
            val cx = out[0][i]
            val cy = out[1][i]
            val w = out[2][i]
            val h = out[3][i]
            // 640-space -> tira o padding e desfaz a escala -> coords da tela
            val x1 = ((cx - w / 2f) - padX) / scale
            val y1 = ((cy - h / 2f) - padY) / scale
            val x2 = ((cx + w / 2f) - padX) / scale
            val y2 = ((cy + h / 2f) - padY) / scale
            candidates.add(
                Detection(
                    x1.coerceIn(0f, srcW.toFloat()),
                    y1.coerceIn(0f, srcH.toFloat()),
                    x2.coerceIn(0f, srcW.toFloat()),
                    y2.coerceIn(0f, srcH.toFloat()),
                    score
                )
            )
        }
        return nms(candidates, iouThreshold)
    }

    private fun nms(boxes: List<Detection>, iouThr: Float): List<Detection> {
        val sorted = boxes.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep.add(best)
            sorted.removeAll { iou(best, it) > iouThr }
        }
        return keep
    }

    private fun iou(a: Detection, b: Detection): Float {
        val interX1 = max(a.x1, b.x1)
        val interY1 = max(a.y1, b.y1)
        val interX2 = min(a.x2, b.x2)
        val interY2 = min(a.y2, b.y2)
        val interW = max(0f, interX2 - interX1)
        val interH = max(0f, interY2 - interY1)
        val inter = interW * interH
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        session.close()
    }
}
