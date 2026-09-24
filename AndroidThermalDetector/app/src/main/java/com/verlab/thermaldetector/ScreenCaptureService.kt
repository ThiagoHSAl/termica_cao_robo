package com.verlab.thermaldetector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.WindowManager

/**
 * Serviço em primeiro plano que:
 *  1. captura a tela via MediaProjection,
 *  2. roda o YOLO em cada frame,
 *  3. desenha as caixas no OverlayView (por cima do Thermal Master).
 */
class ScreenCaptureService : Service() {

    companion object {
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"
        const val ACTION_STOP = "stop"
        private const val CHANNEL_ID = "thermal_capture"
        private const val NOTIF_ID = 1
    }

    private lateinit var yolo: YoloOnnx
    private lateinit var windowManager: WindowManager
    private lateinit var overlay: OverlayView

    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var captureThread: HandlerThread? = null
    private var captureHandler: Handler? = null

    private var screenW = 0
    private var screenH = 0
    private var density = 0
    @Volatile private var busy = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopEverything()
            return START_NOT_STICKY
        }

        startForeground(NOTIF_ID, buildNotification())

        val resultCode = intent!!.getIntExtra(EXTRA_RESULT_CODE, 0)
        val data = intent.getParcelableExtra<Intent>(EXTRA_DATA)!!

        yolo = YoloOnnx(this)

        val metrics = DisplayMetrics()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        screenW = metrics.widthPixels
        screenH = metrics.heightPixels
        density = metrics.densityDpi

        addOverlay()

        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projection = mpm.getMediaProjection(resultCode, data).apply {
            // Obrigatório no Android 14+: registrar callback antes de usar.
            registerCallback(object : MediaProjection.Callback() {
                override fun onStop() { stopEverything() }
            }, null)
        }

        startCapture()
        return START_STICKY
    }

    private fun addOverlay() {
        overlay = OverlayView(this)
        val type = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        )
        params.gravity = Gravity.TOP or Gravity.START
        windowManager.addView(overlay, params)
    }

    private fun startCapture() {
        captureThread = HandlerThread("capture").also { it.start() }
        captureHandler = Handler(captureThread!!.looper)

        imageReader = ImageReader.newInstance(
            screenW, screenH, PixelFormat.RGBA_8888, 2
        )
        virtualDisplay = projection!!.createVirtualDisplay(
            "thermal-capture",
            screenW, screenH, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader!!.surface, null, captureHandler
        )

        imageReader!!.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            if (busy) { image.close(); return@setOnImageAvailableListener }
            busy = true
            try {
                val plane = image.planes[0]
                val buffer = plane.buffer
                val pixelStride = plane.pixelStride
                val rowStride = plane.rowStride
                val rowPadding = rowStride - pixelStride * screenW

                val bmp = Bitmap.createBitmap(
                    screenW + rowPadding / pixelStride, screenH,
                    Bitmap.Config.ARGB_8888
                )
                bmp.copyPixelsFromBuffer(buffer)
                val frame = Bitmap.createBitmap(bmp, 0, 0, screenW, screenH)
                bmp.recycle()

                val dets = yolo.detect(frame)
                frame.recycle()
                val info = "${yolo.lastInferenceMs} ms  |  ${dets.size} pessoa(s)"
                overlay.setDetections(dets, info)
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                image.close()
                busy = false
            }
        }, captureHandler)
    }

    private fun stopEverything() {
        try { virtualDisplay?.release() } catch (_: Exception) {}
        try { imageReader?.close() } catch (_: Exception) {}
        try { projection?.stop() } catch (_: Exception) {}
        try { if (::overlay.isInitialized) windowManager.removeView(overlay) } catch (_: Exception) {}
        try { if (::yolo.isInitialized) yolo.close() } catch (_: Exception) {}
        captureThread?.quitSafely()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        stopEverything()
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID, "Detecção térmica",
                    NotificationManager.IMPORTANCE_LOW
                )
            )
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Detector térmico ativo")
            .setContentText("Capturando a tela e detectando pessoas")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
    }
}
