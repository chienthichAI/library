import { useState, useRef, useEffect, useCallback } from 'react'
import './FaceCapture.css'

const FACE_POSITIONS = [
    { id: 'front', label: 'Nhìn thẳng', instruction: 'Đặt khuôn mặt vào khung oval, nhìn thẳng vào camera' },
    { id: 'left', label: 'Nghiêng trái', instruction: 'Xoay mặt sang trái khoảng 15 độ' },
    { id: 'right', label: 'Nghiêng phải', instruction: 'Xoay mặt sang phải khoảng 15 độ' }
]

const API_URL = 'http://localhost:8000/api/v1'
const AUTO_CAPTURE_DELAY = 2000
const AUTO_CAPTURE_MIN_QUALITY = 0.6
const CHECK_INTERVAL = 800

export default function FaceCapture({ onCapture, requiredCaptures = 3 }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const overlayRef = useRef(null) // Canvas để vẽ bbox
    const streamRef = useRef(null)
    const autoCaptureTimerRef = useRef(null)
    const progressIntervalRef = useRef(null) // Fix memory leak
    const captureImageRef = useRef(null) // Fix stale closure
    const isCheckingRef = useRef(false) // Fix race condition

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

    // --- Camera ---
    useEffect(() => {
        startCamera()
        return () => {
            stopCamera()
            clearTimeout(autoCaptureTimerRef.current)
            clearInterval(progressIntervalRef.current)
            // Revoke object URLs (fix memory leak)
            setCapturedImages(prev => {
                prev.forEach(img => URL.revokeObjectURL(img.url))
                return prev
            })
        }
    }, [])

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
    const drawFaceBoxes = useCallback((faces, videoW, videoH) => {
        const overlay = overlayRef.current
        if (!overlay) return
        overlay.width = videoW
        overlay.height = videoH
        const ctx = overlay.getContext('2d')
        ctx.clearRect(0, 0, videoW, videoH)

        faces.forEach((face, idx) => {
            const { x1, y1, x2, y2, is_primary, confidence } = face
            const isPrimary = is_primary || idx === 0

            // Màu theo primary/secondary
            const color = isPrimary ? '#00e5ff' : 'rgba(255,255,255,0.4)'
            const glowColor = isPrimary ? 'rgba(0,229,255,0.6)' : 'rgba(255,255,255,0.2)'

            // Glow shadow
            ctx.shadowColor = glowColor
            ctx.shadowBlur = isPrimary ? 16 : 6

            // Vẽ góc thay vì full rect (đẹp hơn)
            const cornerLen = Math.min((x2 - x1), (y2 - y1)) * 0.22
            const lw = isPrimary ? 3 : 1.5
            ctx.strokeStyle = color
            ctx.lineWidth = lw
            ctx.lineCap = 'round'

            // Top-left
            ctx.beginPath(); ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1); ctx.stroke()
            // Top-right
            ctx.beginPath(); ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen); ctx.stroke()
            // Bottom-left
            ctx.beginPath(); ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2); ctx.stroke()
            // Bottom-right
            ctx.beginPath(); ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cornerLen, y2); ctx.stroke()

            // Label confidence
            ctx.shadowBlur = 0
            if (isPrimary) {
                const label = `✓ Chính  ${Math.round(confidence * 100)}%`
                ctx.font = 'bold 13px Inter, sans-serif'
                ctx.fillStyle = '#00e5ff'
                const tw = ctx.measureText(label).width
                ctx.fillStyle = 'rgba(0,0,0,0.65)'
                ctx.fillRect(x1, y1 - 26, tw + 12, 22)
                ctx.fillStyle = '#00e5ff'
                ctx.fillText(label, x1 + 6, y1 - 9)
            } else {
                const label = `${Math.round(confidence * 100)}%`
                ctx.font = '11px Inter, sans-serif'
                ctx.fillStyle = 'rgba(255,255,255,0.6)'
                ctx.fillText(label, x1 + 4, y1 - 6)
            }
        })
    }, [])

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
        if (!isStreaming) return
        const interval = setInterval(checkQuality, CHECK_INTERVAL)
        return () => clearInterval(interval)
    }, [isStreaming, checkQuality])

    // --- Capture ---
    const captureImage = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) return
        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)
        ctx.drawImage(video, 0, 0)

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

                {/* Oval guide */}
                <div className="fc-oval-guide">
                    <div className="fc-oval-ring" />
                    <div className="fc-scan-line" />
                </div>

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
