package com.verlab.thermaldetector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/**
 * View transparente, em tela cheia, que desenha as caixas de detecção
 * POR CIMA de qualquer app (ex.: o Thermal Master). É isso que faz tudo
 * aparecer em UMA tela só.
 */
class OverlayView(context: Context) : View(context) {

    private var detections: List<Detection> = emptyList()
    private var infoText: String = ""

    private val boxPaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        textSize = 42f
        isAntiAlias = true
    }
    private val textBg = Paint().apply {
        color = Color.parseColor("#99000000")
        style = Paint.Style.FILL
    }

    fun setDetections(dets: List<Detection>, info: String) {
        detections = dets
        infoText = info
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Info no topo (tempo de inferência + nº de pessoas)
        if (infoText.isNotEmpty()) {
            val tw = textPaint.measureText(infoText)
            canvas.drawRect(0f, 0f, tw + 24f, 60f, textBg)
            canvas.drawText(infoText, 12f, 44f, textPaint)
        }
        for (d in detections) {
            canvas.drawRect(d.x1, d.y1, d.x2, d.y2, boxPaint)
            val label = "person ${"%.2f".format(d.score)}"
            val tw = textPaint.measureText(label)
            val ty = if (d.y1 > 46f) d.y1 else d.y2
            canvas.drawRect(d.x1, ty - 46f, d.x1 + tw + 12f, ty, textBg)
            canvas.drawText(label, d.x1 + 6f, ty - 10f, textPaint)
        }
    }
}
