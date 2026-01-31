import { useState, useRef, useEffect, useCallback } from 'react'
import './FaceCapture.css'

const FACE_POSITIONS = [
    { id: 'front', label: 'Nhìn thẳng', instruction: 'Đặt khuôn mặt vào khung tròn, nhìn thẳng vào camera' },
    { id: 'left', label: 'Nghiêng trái', instruction: 'Xoay mặt sang trái khoảng 15 độ' },
    { id: 'right', label: 'Nghiêng phải', instruction: 'Xoay mặt sang phải khoảng 15 độ' }
]

const API_URL = 'http://localhost:8000/api/v1'

// Auto-capture settings
const AUTO_CAPTURE_DELAY = 2000 // 2 seconds of stable quality before auto-capture
const AUTO_CAPTURE_MIN_QUALITY = 0.6 // Minimum quality score for auto-capture

export default function FaceCapture({ onCapture, requiredCaptures = 3 }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const streamRef = useRef(null)
    const autoCaputreTimerRef = useRef(null) // Timer for auto-capture

    const [isStreaming, setIsStreaming] = useState(false)
    const [currentPosition, setCurrentPosition] = useState(0)
    const [capturedImages, setCapturedImages] = useState([])
    const [faceStatus, setFaceStatus] = useState('waiting') // waiting, valid, invalid, checking
    const [countdown, setCountdown] = useState(null)
    const [statusMessage, setStatusMessage] = useState('Đang khởi động camera...')
    const [qualityScore, setQualityScore] = useState(null)
    const [qualityIssues, setQualityIssues] = useState([])
    const [autoCaputreProgress, setAutoCaputreProgress] = useState(0) // 0-100% progress to auto-capture

    // Start camera
    useEffect(() => {
        startCamera()
        return () => {
            stopCamera()
            // Clear auto-capture timer on unmount
            if (autoCaputreTimerRef.current) {
                clearTimeout(autoCaputreTimerRef.current)
            }
        }
    }, [])

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            })

            streamRef.current = stream

            if (videoRef.current) {
                videoRef.current.srcObject = stream
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current.play()
                    setIsStreaming(true)
                    setStatusMessage('Đưa khuôn mặt vào khung tròn')
                }
            }
        } catch (err) {
            console.error('Camera error:', err)
            setStatusMessage('Không thể truy cập camera. Vui lòng cấp quyền.')
        }
    }

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
        }
    }

    // Real-time quality check using backend API
    const checkQuality = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current || !isStreaming || faceStatus === 'checking' || countdown !== null) return

        const video = videoRef.current
        const canvas = canvasRef.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d').drawImage(video, 0, 0)

        setFaceStatus('checking')

        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData()
                formData.append('image', blob, 'quality_check.jpg')

                const response = await fetch(`${API_URL}/auth/check-quality`, {
                    method: 'POST',
                    body: formData
                })

                if (response.ok) {
                    const result = await response.json()
                    setQualityScore(result.overall_score)
                    setQualityIssues(result.issues || [])

                    if (result.is_valid && result.overall_score >= AUTO_CAPTURE_MIN_QUALITY) {
                        setFaceStatus('valid')
                        setStatusMessage('✓ Giữ nguyên! Đang tự động chụp...')

                        // Start auto-capture timer if not already running
                        if (!autoCaputreTimerRef.current) {
                            setAutoCaputreProgress(0)

                            // Progress animation
                            let progress = 0
                            const progressInterval = setInterval(() => {
                                progress += 10
                                setAutoCaputreProgress(progress)
                                if (progress >= 100) {
                                    clearInterval(progressInterval)
                                }
                            }, AUTO_CAPTURE_DELAY / 10)

                            // Auto-capture after delay
                            autoCaputreTimerRef.current = setTimeout(() => {
                                clearInterval(progressInterval)
                                setAutoCaputreProgress(100)
                                triggerAutoCapture()
                                autoCaputreTimerRef.current = null
                            }, AUTO_CAPTURE_DELAY)
                        }
                    } else {
                        setFaceStatus('invalid')
                        setStatusMessage(result.message || 'Điều chỉnh vị trí khuôn mặt')

                        // Cancel auto-capture if quality drops
                        if (autoCaputreTimerRef.current) {
                            clearTimeout(autoCaputreTimerRef.current)
                            autoCaputreTimerRef.current = null
                            setAutoCaputreProgress(0)
                        }
                    }
                } else {
                    const errorData = await response.json().catch(() => ({}));
                    setFaceStatus('invalid')
                    setStatusMessage(errorData.detail || 'Lỗi kiểm tra chất lượng. Thử lại.')

                    // Cancel auto-capture on error
                    if (autoCaputreTimerRef.current) {
                        clearTimeout(autoCaputreTimerRef.current)
                        autoCaputreTimerRef.current = null
                        setAutoCaputreProgress(0)
                    }
                }
            } catch (err) {
                console.log('Quality API error:', err)
                setFaceStatus('valid') // Fallback to allow capture
                setStatusMessage('Đưa khuôn mặt vào khung')
            }
        }, 'image/jpeg', 0.8)
    }, [isStreaming, countdown])

    // Periodic quality check
    useEffect(() => {
        if (!isStreaming) return

        const interval = setInterval(checkQuality, 800) // Check every 800ms for smoother auto-capture
        return () => clearInterval(interval)
    }, [isStreaming, checkQuality])

    // Capture image function
    const captureImage = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) return

        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)

        canvas.toBlob((blob) => {
            const imageUrl = URL.createObjectURL(blob)
            const newImage = {
                id: FACE_POSITIONS[currentPosition].id,
                url: imageUrl,
                blob: blob,
                position: currentPosition,
                qualityScore: qualityScore
            }

            const newCaptured = [...capturedImages, newImage]
            setCapturedImages(newCaptured)

            if (newCaptured.length >= requiredCaptures) {
                // All captures complete
                setFaceStatus('checking')
                setStatusMessage('⌛ Đang xử lý ảnh cuối cùng...')

                // Use a local reference to ensure we pass the correct array
                const finalImages = [...newCaptured];
                setTimeout(() => {
                    onCapture(finalImages)
                }, 500)
            } else {
                // Move to next position
                const nextPos = currentPosition + 1;
                setCurrentPosition(nextPos)
                setFaceStatus('waiting')
                setCountdown(null)
                setStatusMessage(FACE_POSITIONS[nextPos]?.instruction || '')
            }
        }, 'image/jpeg', 0.9)
    }, [currentPosition, capturedImages, requiredCaptures, onCapture, qualityScore])

    // Auto-capture trigger function
    const triggerAutoCapture = useCallback(() => {
        // Reset progress and start quick countdown
        setAutoCaputreProgress(0)
        setCountdown(3)

        const countdownInterval = setInterval(() => {
            setCountdown(prev => {
                if (prev <= 1) {
                    clearInterval(countdownInterval)
                    captureImage()
                    return null
                }
                return prev - 1
            })
        }, 800) // Faster countdown for better UX
    }, [captureImage])

    const handleCaptureClick = () => {
        if (faceStatus === 'invalid' && qualityIssues.includes('no_face')) {
            // Only strictly block if no face at all
            setStatusMessage(`⚠️ Không thấy mặt: ${statusMessage}`);
            return
        }

        if (faceStatus === 'waiting' && countdown === null) {
            // Force a quality check before allowing capture if waiting
            checkQuality();
        }

        // Start countdown
        setCountdown(3)

        const countdownInterval = setInterval(() => {
            setCountdown(prev => {
                if (prev <= 1) {
                    clearInterval(countdownInterval)
                    captureImage()
                    return null
                }
                return prev - 1
            })
        }, 1000)
    }

    const currentInstruction = FACE_POSITIONS[currentPosition]

    return (
        <div className="face-capture animate-fade-in">
            <div className="capture-header">
                <h3>{currentInstruction?.label}</h3>
                <p>{currentInstruction?.instruction}</p>
            </div>

            {/* Camera view */}
            <div className="camera-container">
                <video
                    ref={videoRef}
                    className="camera-video"
                    autoPlay
                    playsInline
                    muted
                />

                {/* Face guide overlay */}
                <div className={`face-guide-overlay ${faceStatus}`}>
                    <div className="face-oval">
                        <div className="scan-line"></div>
                    </div>
                    <div className="guide-corners">
                        <span className="corner tl"></span>
                        <span className="corner tr"></span>
                        <span className="corner bl"></span>
                        <span className="corner br"></span>
                    </div>
                </div>

                {/* Quality indicator */}
                {qualityScore !== null && (
                    <div className={`quality-indicator ${faceStatus}`}>
                        <div className="quality-bar">
                            <div
                                className="quality-fill"
                                style={{ width: `${qualityScore * 100}%` }}
                            />
                        </div>
                        <span>{Math.round(qualityScore * 100)}%</span>
                    </div>
                )}

                {/* Auto-capture progress overlay */}
                {autoCaputreProgress > 0 && autoCaputreProgress < 100 && !countdown && (
                    <div className="auto-capture-progress">
                        <div className="progress-ring">
                            <svg viewBox="0 0 100 100">
                                <circle
                                    className="progress-ring-bg"
                                    cx="50"
                                    cy="50"
                                    r="45"
                                />
                                <circle
                                    className="progress-ring-fill"
                                    cx="50"
                                    cy="50"
                                    r="45"
                                    style={{
                                        strokeDasharray: `${autoCaputreProgress * 2.83} 283`,
                                        transform: 'rotate(-90deg)',
                                        transformOrigin: 'center'
                                    }}
                                />
                            </svg>
                        </div>
                        <span className="auto-capture-text">Giữ yên...</span>
                    </div>
                )}

                {/* Countdown overlay */}
                {countdown && (
                    <div className="countdown-overlay">
                        <span className="countdown-number">{countdown}</span>
                    </div>
                )}

                {/* Hidden canvas for capture */}
                <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>

            {/* Status message */}
            <div className={`status-message ${faceStatus}`}>
                <span className="status-icon">
                    {faceStatus === 'valid' ? '✓' : faceStatus === 'invalid' ? '!' : faceStatus === 'checking' ? '⟳' : '○'}
                </span>
                {statusMessage}
            </div>

            {/* Quality issues */}
            {qualityIssues.length > 0 && faceStatus === 'invalid' && (
                <div className="quality-issues">
                    {qualityIssues.slice(0, 2).map((issue, idx) => (
                        <span key={idx} className="issue-tag">
                            {issue === 'too_dark' && '🌙 Quá tối'}
                            {issue === 'too_bright' && '☀️ Quá sáng'}
                            {issue === 'blurry' && '📷 Mờ'}
                            {issue === 'face_too_small' && '👤 Mặt nhỏ'}
                            {issue === 'face_not_centered' && '↔️ Chưa căn giữa'}
                            {issue === 'multiple_faces' && '👥 Nhiều mặt'}
                        </span>
                    ))}
                </div>
            )}

            {/* Progress indicators */}
            <div className="capture-progress">
                {FACE_POSITIONS.map((pos, idx) => (
                    <div
                        key={pos.id}
                        className={`progress-dot ${idx < capturedImages.length ? 'captured' : ''} ${idx === currentPosition ? 'current' : ''}`}
                    >
                        {idx < capturedImages.length ? (
                            <img src={capturedImages[idx]?.url} alt={pos.label} />
                        ) : (
                            <span>{idx + 1}</span>
                        )}
                        <label>{pos.label}</label>
                    </div>
                ))}
            </div>

            {/* Manual capture button (backup option) */}
            <button
                className={`btn btn-secondary btn-small capture-btn-manual ${faceStatus === 'valid' ? 'ready' : ''}`}
                onClick={handleCaptureClick}
                disabled={!isStreaming || countdown !== null}
            >
                {countdown ? `Chụp sau ${countdown}...` : '📸 Chụp thủ công'}
            </button>

            <p className="capture-hint">
                🤖 <strong>Tự động chụp</strong> khi khuôn mặt đủ điều kiện • Giữ yên trong 2 giây
            </p>
        </div>
    )
}
