import { useState, useRef, useEffect, useCallback } from 'react'
import { FilesetResolver, FaceDetector } from '@mediapipe/tasks-vision'
import { API_URL } from '../config'
import './FaceCapture.css'

const FACE_POSITIONS = [
    { id: 'front', label: 'Nhìn thẳng', instruction: 'Đặt khuôn mặt vào khung oval, nhìn thẳng vào camera' },
    { id: 'left', label: 'Nghiêng trái', instruction: 'Xoay mặt sang trái khoảng 15 độ' },
    { id: 'right', label: 'Nghiêng phải', instruction: 'Xoay mặt sang phải khoảng 15 độ' }
]

const AUTO_CAPTURE_DELAY = 2000
const AUTO_CAPTURE_MIN_QUALITY = 0.6
const CHECK_INTERVAL = 800

export default function FaceCapture({ onCapture, requiredCaptures = 3 }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const overlayRef = useRef(null) // Canvas để vẽ bbox
    const streamRef = useRef(null)
    const smoothedBboxRef = useRef(null)
    const autoCaptureTimerRef = useRef(null)
    const progressIntervalRef = useRef(null) 
    const captureImageRef = useRef(null) 
    const isCheckingRef = useRef(false) 
    const rafRef = useRef(null) 
    
    // WebSockets & MediaPipe refs
    const faceDetectorRef = useRef(null)
    const wsRef = useRef(null)
    const lastFrameTimeRef = useRef(0)

    const [isStreaming, setIsStreaming] = useState(false)
    const [currentPosition, setCurrentPosition] = useState(0)
    const [capturedImages, setCapturedImages] = useState([])
    const [faceStatus, setFaceStatus] = useState('waiting')
    const [countdown, setCountdown] = useState(null)
    const [statusMessage, setStatusMessage] = useState('Đang khởi động camera...')
    const [qualityScore, setQualityScore] = useState(null)
    const [qualityIssues, setQualityIssues] = useState([])
    const [autoCaptureProgress, setAutoCaptureProgress] = useState(0)
    const [cooldown, setCooldown] = useState(false)
    const [detectedFaces, setDetectedFaces] = useState([]) // Bbox data từ API
    const [trackingStatus, setTrackingStatus] = useState('searching') // 'searching' | 'tracking'

    // --- Camera ---
    useEffect(() => {
        initFaceDetector().then(() => {
            connectWebSocket()
            startCamera()
        })
        return () => {
            stopCamera()
            clearTimeout(autoCaptureTimerRef.current)
            clearInterval(progressIntervalRef.current)
            if (wsRef.current) wsRef.current.close()
            if (rafRef.current) cancelAnimationFrame(rafRef.current)
            // Revoke object URLs
            setCapturedImages(prev => {
                prev.forEach(img => URL.revokeObjectURL(img.url))
                return prev
            })
        }
    }, [])
    
    const initFaceDetector = async () => {
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
            )
            faceDetectorRef.current = await FaceDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                    delegate: "GPU" // Offload scanning to GPU for high perf
                },
                runningMode: "VIDEO"
            })
        } catch (err) {
            console.error("Failed to init FaceDetector:", err)
        }
    }

    const connectWebSocket = () => {
        const wsUrl = API_URL.replace(/^http/, 'ws') + '/auth/ws/stream'
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        
        ws.onmessage = (event) => {
            isCheckingRef.current = false
            let result
            try {
                result = JSON.parse(event.data)
            } catch (err) {
                console.warn('Invalid WebSocket payload:', err)
                return
            }
            
            setQualityScore(result.quality_score)
            setQualityIssues(result.quality_issues || [])
            
            // Wait until components are ready
            if (countdown !== null || cooldown) return;
            
            const isFaceMissingError = result.error_message?.toLowerCase().includes('khuôn mặt')
            const isValid = result.quality_score >= AUTO_CAPTURE_MIN_QUALITY && (!result.quality_issues || result.quality_issues.length === 0) && !isFaceMissingError
            
            if (isValid) {
                setFaceStatus('valid')
                setStatusMessage('✓ Giữ nguyên! Đang tự động chụp...')
                if (!autoCaptureTimerRef.current) {
                    setAutoCaptureProgress(0)
                    let progress = 0
                    progressIntervalRef.current = setInterval(() => {
                        progress += 10
                        setAutoCaptureProgress(progress)
                        if (progress >= 100) clearInterval(progressIntervalRef.current)
                    }, AUTO_CAPTURE_DELAY / 10)

                    autoCaptureTimerRef.current = setTimeout(() => {
                        clearInterval(progressIntervalRef.current)
                        setAutoCaptureProgress(100)
                        triggerAutoCapture()
                        autoCaptureTimerRef.current = null
                    }, AUTO_CAPTURE_DELAY)
                }
            } else {
                setFaceStatus('invalid')
                setStatusMessage(result.error_message || 'Điều chỉnh vị trí khuôn mặt')
                clearTimeout(autoCaptureTimerRef.current)
                clearInterval(progressIntervalRef.current)
                autoCaptureTimerRef.current = null
                setAutoCaptureProgress(0)
            }
        }
    }

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
            })
            streamRef.current = stream
            if (videoRef.current) {
                videoRef.current.srcObject = stream
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current.play().catch(e => console.error('Video play error:', e))
                    setIsStreaming(true)
                    setStatusMessage('Đưa khuôn mặt vào khung oval')
                }
            }
        } catch {
            setStatusMessage('Không thể truy cập camera. Vui lòng cấp quyền.')
        }
    }

    const stopCamera = () => {
        streamRef.current?.getTracks().forEach(t => t.stop())
    }

    // --- Vẽ bbox lên overlay canvas ---
    const drawFaceBoxes = useCallback((detections) => {
        const overlay = overlayRef.current
        const video = videoRef.current
        if (!overlay || !video) return

        // Sync overlay size with video display size
        const displayWidth = video.clientWidth
        const displayHeight = video.clientHeight
        if (overlay.width !== displayWidth || overlay.height !== displayHeight) {
            overlay.width = displayWidth
            overlay.height = displayHeight
        }

        const ctx = overlay.getContext('2d')
        ctx.clearRect(0, 0, overlay.width, overlay.height)

        const scaleX = overlay.width / video.videoWidth
        const scaleY = overlay.height / video.videoHeight

        detections.forEach((det, idx) => {
            const isPrimary = idx === 0
            if (!isPrimary) return

            const confidence = det.categories[0].score
            const { originX, width, originY, height } = det.boundingBox

            // 1. Calculate Target Mirrored Coords
            const targetX1 = overlay.width - ((originX + width) * scaleX)
            const targetY1 = originY * scaleY
            const targetW = width * scaleX
            const targetH = height * scaleY

            // 2. Smooth the Bounding Box (LERP)
            if (!smoothedBboxRef.current) {
                smoothedBboxRef.current = { x: targetX1, y: targetY1, w: targetW, h: targetH }
            } else {
                const lerp = 0.25
                smoothedBboxRef.current.x += (targetX1 - smoothedBboxRef.current.x) * lerp
                smoothedBboxRef.current.y += (targetY1 - smoothedBboxRef.current.y) * lerp
                smoothedBboxRef.current.w += (targetW - smoothedBboxRef.current.w) * lerp
                smoothedBboxRef.current.h += (targetH - smoothedBboxRef.current.h) * lerp
            }

            const { x, y, w, h } = smoothedBboxRef.current
            const x2 = x + w
            const y2 = y + h

            const color = '#00e5ff'
            const glowColor = 'rgba(0,229,255,0.4)'
            const now = Date.now() / 1000

            const centerX = x + w/2
            const centerY = y + h/2
            
            // --- DRAW HUD BRACKETS ---
            ctx.save()
            
            // Subtle Background fill
            ctx.fillStyle = 'rgba(0, 229, 255, 0.05)'
            ctx.fillRect(x, y, w, h)

            // Inner Corner Highlights
            ctx.shadowColor = color
            ctx.shadowBlur = 10
            ctx.strokeStyle = color
            ctx.lineWidth = 3
            ctx.lineCap = 'butt'

            const brSize = w * 0.15 // Bracket size
            // Corners
            ctx.beginPath(); ctx.moveTo(x, y + brSize); ctx.lineTo(x, y); ctx.lineTo(x + brSize, y); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(x2 - brSize, y); ctx.lineTo(x2, y); ctx.lineTo(x2, y + brSize); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(x, y2 - brSize); ctx.lineTo(x, y2); ctx.lineTo(x + brSize, y2); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(x2 - brSize, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - brSize); ctx.stroke()

            // Side Ticks
            ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(x - 5, centerY); ctx.lineTo(x + 5, centerY); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(x2 - 5, centerY); ctx.lineTo(x2 + 5, centerY); ctx.stroke()

            ctx.restore()

            // --- DATA READOUTS ---
            const fontSize = 11
            ctx.font = `${fontSize}px "JetBrains Mono", "Courier New", monospace`
            ctx.fillStyle = color
            
            ctx.fillText(`STATUS: ANALYZING`, x, y - 12)
            
            const metrics = [
                `VALIDITY: ${faceStatus.toUpperCase()}`,
                `CONF: ${Math.round(confidence * 100)}%`,
                `LOC: [${Math.round(x)},${Math.round(y)}]`
            ]
            
            metrics.forEach((m, i) => {
                ctx.fillStyle = 'rgba(0,229,255,0.8)'
                ctx.fillText(m, x2 + 10, y + (i * (fontSize + 6)) + 15)
            })

            // Scanning Line
            const scanPos = (Math.sin(now * 3) + 1) / 2
            const scanY = y + (h * scanPos)
            const grad = ctx.createLinearGradient(x, scanY, x2, scanY)
            grad.addColorStop(0, 'transparent')
            grad.addColorStop(0.5, 'rgba(0,229,255,0.3)')
            grad.addColorStop(1, 'transparent')
            ctx.fillStyle = grad
            ctx.fillRect(x, scanY - 1, w, 2)
        })
    }, [faceStatus])

    // --- Quality check + face detection ---
    const checkQuality = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current || !isStreaming
            || isCheckingRef.current || countdown !== null || cooldown) return

        const video = videoRef.current
        const canvas = canvasRef.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        const ctx = canvas.getContext('2d')
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)
        ctx.drawImage(video, 0, 0)

        isCheckingRef.current = true

        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData()
                formData.append('image', blob, 'quality_check.jpg')
                const response = await fetch(`${API_URL}/auth/check-quality`, {
                    method: 'POST', body: formData
                })

                if (response.ok) {
                    const result = await response.json()
                    setQualityScore(result.overall_score)
                    setQualityIssues(result.issues || [])

                    // Vẽ bbox nếu API trả về faces
                    if (result.faces && result.faces.length > 0) {
                        setDetectedFaces(result.faces)
                        drawFaceBoxes(result.faces, video.videoWidth, video.videoHeight)
                    } else {
                        setDetectedFaces([])
                        const overlay = overlayRef.current
                        if (overlay) {
                            const c = overlay.getContext('2d')
                            c.clearRect(0, 0, overlay.width, overlay.height)
                        }
                    }

                    if (result.is_valid && result.overall_score >= AUTO_CAPTURE_MIN_QUALITY) {
                        setFaceStatus('valid')
                        setStatusMessage('✓ Giữ nguyên! Đang tự động chụp...')
                        if (!autoCaptureTimerRef.current) {
                            setAutoCaptureProgress(0)
                            let progress = 0
                            progressIntervalRef.current = setInterval(() => {
                                progress += 10
                                setAutoCaptureProgress(progress)
                                if (progress >= 100) clearInterval(progressIntervalRef.current)
                            }, AUTO_CAPTURE_DELAY / 10)

                            autoCaptureTimerRef.current = setTimeout(() => {
                                clearInterval(progressIntervalRef.current)
                                setAutoCaptureProgress(100)
                                triggerAutoCapture()
                                autoCaptureTimerRef.current = null
                            }, AUTO_CAPTURE_DELAY)
                        }
                    } else {
                        setFaceStatus('invalid')
                        setStatusMessage(result.message || 'Điều chỉnh vị trí khuôn mặt')
                        clearTimeout(autoCaptureTimerRef.current)
                        clearInterval(progressIntervalRef.current)
                        autoCaptureTimerRef.current = null
                        setAutoCaptureProgress(0)
                    }
                } else {
                    setFaceStatus('waiting')
                    clearTimeout(autoCaptureTimerRef.current)
                    autoCaptureTimerRef.current = null
                    setAutoCaptureProgress(0)
                }
            } catch {
                setFaceStatus('valid')
                setStatusMessage('Đưa khuôn mặt vào khung')
            } finally {
                isCheckingRef.current = false // Luôn reset
            }
        }, 'image/jpeg', 0.8)
    }, [isStreaming, countdown, cooldown, drawFaceBoxes])

    useEffect(() => {
        if (isStreaming) {
            rafRef.current = requestAnimationFrame(trackingLoop)
        }
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current)
        }
    }, [isStreaming, trackingLoop])

    // --- Capture ---
    const captureImage = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) return
        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        
        // Scale down to 640x360 for faster processing and smaller payloads
        canvas.width = 640
        canvas.height = 360
        ctx.drawImage(video, 0, 0, 640, 360)

        canvas.toBlob((blob) => {
            const imageUrl = URL.createObjectURL(blob)
            const newImage = {
                id: FACE_POSITIONS[currentPosition].id,
                url: imageUrl,
                blob,
                position: currentPosition,
                qualityScore
            }
            const newCaptured = [...capturedImages, newImage]
            setCapturedImages(newCaptured)

            if (newCaptured.length >= requiredCaptures) {
                setFaceStatus('checking')
                setStatusMessage('⌛ Đang xử lý...')
                const finalImages = [...newCaptured]
                setTimeout(() => onCapture(finalImages), 500)
            } else {
                const nextPos = currentPosition + 1
                setCurrentPosition(nextPos)
                setFaceStatus('waiting')
                setCountdown(null)
                setDetectedFaces([])
                // Reset tracker state for the new position
                setTrackingStatus('searching')
                setCooldown(true)
                setStatusMessage(`Chuẩn bị: ${FACE_POSITIONS[nextPos]?.instruction}`)
                setTimeout(() => {
                    setCooldown(false)
                    setStatusMessage(FACE_POSITIONS[nextPos]?.instruction || '')
                }, 2500)
            }
        }, 'image/jpeg', 0.9)
    }, [currentPosition, capturedImages, requiredCaptures, onCapture, qualityScore])

    // Giữ ref luôn fresh — fix stale closure
    useEffect(() => { captureImageRef.current = captureImage }, [captureImage])

    const triggerAutoCapture = useCallback(() => {
        setAutoCaptureProgress(0)
        setCountdown(3)
        const iv = setInterval(() => {
            setCountdown(prev => {
                if (prev <= 1) {
                    clearInterval(iv)
                    captureImageRef.current?.() // Dùng ref, không phải stale closure
                    return null
                }
                return prev - 1
            })
        }, 800)
    }, [])

    const handleCaptureClick = () => {
        // Fix double capture: clear auto trước khi manual
        if (autoCaptureTimerRef.current) {
            clearTimeout(autoCaptureTimerRef.current)
            clearInterval(progressIntervalRef.current)
            autoCaptureTimerRef.current = null
            setAutoCaptureProgress(0)
        }
        if (faceStatus === 'invalid' && qualityIssues.includes('no_face')) {
            setStatusMessage('⚠️ Không thấy mặt trong khung hình')
            return
        }
        setCountdown(3)
        const iv = setInterval(() => {
            setCountdown(prev => {
                if (prev <= 1) { clearInterval(iv); captureImageRef.current?.(); return null }
                return prev - 1
            })
        }, 1000)
    }

    const currentInstruction = FACE_POSITIONS[currentPosition]

    return (
        <div className="fc-wrapper animate-fade-in">
            {/* Header */}
            <div className="fc-header">
                <div className="fc-logo">🏛️ SmartLib</div>
                <h2 className="fc-title">Xác thực khuôn mặt</h2>
                <p className="fc-subtitle">{currentInstruction?.instruction}</p>
            </div>

            {/* Camera */}
            <div className={`fc-camera-wrap ${faceStatus}`}>
                <video ref={videoRef} className="fc-video" autoPlay playsInline muted />

                {/* Bbox overlay */}
                <canvas ref={overlayRef} className="fc-overlay" />

                {/* Tracking status badge */}
                <div className={`fc-tracker-badge ${trackingStatus}`}>
                    {trackingStatus === 'tracking' ? (
                        <>
                            <span className="fc-tracker-dot" />
                            <span>Đang theo dõi</span>
                            {detectedFaces.length > 1 && (
                                <span className="fc-multi-face-badge">👥 {detectedFaces.length} mặt</span>
                            )}
                        </>
                    ) : (
                        <>
                            <span className="fc-tracker-dot" />
                            <span>Đang tìm...</span>
                        </>
                    )}
                </div>

                {/* Removed fc-oval-guide as requested */}


                {/* Countdown */}
                {countdown && (
                    <div className="fc-countdown">
                        <span>{countdown}</span>
                    </div>
                )}

                {/* Auto-capture progress ring */}
                {autoCaptureProgress > 0 && !countdown && (
                    <div className="fc-progress-ring-wrap">
                        <svg viewBox="0 0 100 100" className="fc-progress-svg">
                            <circle className="fc-ring-bg" cx="50" cy="50" r="44" />
                            <circle
                                className="fc-ring-fill"
                                cx="50" cy="50" r="44"
                                style={{
                                    strokeDasharray: `${autoCaptureProgress * 2.76} 276`,
                                    transform: 'rotate(-90deg)',
                                    transformOrigin: 'center'
                                }}
                            />
                        </svg>
                        <span className="fc-hold-text">Giữ yên</span>
                    </div>
                )}

                {/* Corner decorations */}
                <span className="fc-corner tl" /><span className="fc-corner tr" />
                <span className="fc-corner bl" /><span className="fc-corner br" />

                <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>

            {/* Quality bar */}
            {qualityScore !== null && (
                <div className="fc-quality-bar-wrap">
                    <div className="fc-quality-labels">
                        <span>Chất lượng ảnh</span>
                        <span className={`fc-quality-pct ${faceStatus}`}>
                            {Math.round(qualityScore * 100)}%
                        </span>
                    </div>
                    <div className="fc-quality-track">
                        <div
                            className={`fc-quality-fill ${faceStatus}`}
                            style={{ width: `${qualityScore * 100}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Status */}
            <div className={`fc-status ${faceStatus}`}>
                <span className="fc-status-icon">
                    {faceStatus === 'valid' ? '✓'
                        : faceStatus === 'invalid' ? '!'
                            : faceStatus === 'checking' ? '⟳' : '○'}
                </span>
                <span>{statusMessage}</span>
            </div>

            {/* Issues */}
            {qualityIssues.length > 0 && faceStatus === 'invalid' && (
                <div className="fc-issues">
                    {qualityIssues.slice(0, 3).map((issue, i) => (
                        <span key={i} className="fc-issue-tag">
                            {issue === 'too_dark' && '🌙 Quá tối'}
                            {issue === 'too_bright' && '☀️ Quá sáng'}
                            {issue === 'blurry' && '📷 Ảnh mờ'}
                            {issue === 'face_too_small' && '👤 Mặt quá nhỏ'}
                            {issue === 'face_not_centered' && '↔️ Chưa căn giữa'}
                            {issue === 'multiple_faces' && `👥 ${detectedFaces.length} khuôn mặt`}
                            {issue === 'no_face' && '❌ Không thấy mặt'}
                        </span>
                    ))}
                </div>
            )}

            {/* Progress dots */}
            <div className="fc-progress-dots">
                {FACE_POSITIONS.map((pos, idx) => (
                    <div
                        key={pos.id}
                        className={`fc-dot
              ${idx < capturedImages.length ? 'done' : ''}
              ${idx === currentPosition ? 'active' : ''}`}
                    >
                        {idx < capturedImages.length
                            ? <img src={capturedImages[idx]?.url} alt={pos.label} />
                            : <span>{idx + 1}</span>
                        }
                        <label>{pos.label}</label>
                    </div>
                ))}
            </div>

            {/* Manual button */}
            <button
                className={`fc-btn-manual ${faceStatus === 'valid' ? 'ready' : ''}`}
                onClick={handleCaptureClick}
                disabled={!isStreaming || countdown !== null}
            >
                {countdown ? `📸 Chụp sau ${countdown}...` : '📸 Chụp thủ công'}
            </button>

            <p className="fc-hint">
                🤖 <strong>Tự động chụp</strong> khi khuôn mặt đủ điều kiện • Giữ yên 2 giây
            </p>
        </div>
    )
}
