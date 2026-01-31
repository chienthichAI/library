import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_URL } from '../config'
import './VerificationScreen.css'

// Auto-verify settings
const AUTO_VERIFY_DELAY = 1500 // 1.5 seconds of stable quality before auto-verify
const AUTO_VERIFY_MIN_QUALITY = 0.6 // Minimum quality score

export default function VerificationScreen() {
    const navigate = useNavigate()
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const streamRef = useRef(null)
    const autoVerifyTimerRef = useRef(null)

    const [isStreaming, setIsStreaming] = useState(false)
    const [verificationStatus, setVerificationStatus] = useState('idle') // idle, checking, verifying, success, failed
    const [verifiedStudent, setVerifiedStudent] = useState(null)
    const [errorMessage, setErrorMessage] = useState(null)
    const [qualityScore, setQualityScore] = useState(null)
    const [faceStatus, setFaceStatus] = useState('waiting') // waiting, valid, invalid
    const [autoVerifyProgress, setAutoVerifyProgress] = useState(0)
    const [statusMessage, setStatusMessage] = useState('Đang khởi động camera...')

    useEffect(() => {
        startCamera()
        return () => {
            stopCamera()
            if (autoVerifyTimerRef.current) {
                clearTimeout(autoVerifyTimerRef.current)
            }
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
                    videoRef.current.play()
                    setIsStreaming(true)
                    setStatusMessage('Đưa khuôn mặt vào khung tròn')
                }
            }
        } catch (err) {
            setErrorMessage('Không thể truy cập camera')
            setStatusMessage('Không thể truy cập camera')
        }
    }

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => {
                track.stop()
                console.log('Camera track stopped')
            })
            streamRef.current = null
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null
        }
    }

    // Real-time quality check
    const checkQuality = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current || !isStreaming ||
            verificationStatus === 'verifying' || verificationStatus === 'success' ||
            faceStatus === 'checking') return

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

                    if (result.is_valid && result.overall_score >= AUTO_VERIFY_MIN_QUALITY) {
                        setFaceStatus('valid')
                        setStatusMessage('✓ Giữ nguyên! Đang tự động xác thực...')

                        // Start auto-verify timer if not already running
                        if (!autoVerifyTimerRef.current && verificationStatus === 'idle') {
                            setAutoVerifyProgress(0)

                            // Progress animation
                            let progress = 0
                            const progressInterval = setInterval(() => {
                                progress += 10
                                setAutoVerifyProgress(progress)
                                if (progress >= 100) {
                                    clearInterval(progressInterval)
                                }
                            }, AUTO_VERIFY_DELAY / 10)

                            // Auto-verify after delay
                            autoVerifyTimerRef.current = setTimeout(() => {
                                clearInterval(progressInterval)
                                setAutoVerifyProgress(100)
                                handleVerify()
                                autoVerifyTimerRef.current = null
                            }, AUTO_VERIFY_DELAY)
                        }
                    } else {
                        setFaceStatus('invalid')
                        setStatusMessage(result.message || 'Điều chỉnh vị trí khuôn mặt')

                        // Cancel auto-verify if quality drops
                        if (autoVerifyTimerRef.current) {
                            clearTimeout(autoVerifyTimerRef.current)
                            autoVerifyTimerRef.current = null
                            setAutoVerifyProgress(0)
                        }
                    }
                } else {
                    setFaceStatus('invalid')
                    setStatusMessage('Lỗi kiểm tra chất lượng')

                    if (autoVerifyTimerRef.current) {
                        clearTimeout(autoVerifyTimerRef.current)
                        autoVerifyTimerRef.current = null
                        setAutoVerifyProgress(0)
                    }
                }
            } catch (err) {
                console.log('Quality API error:', err)
                setFaceStatus('valid')
                setStatusMessage('Đưa khuôn mặt vào khung')
            }
        }, 'image/jpeg', 0.8)
    }, [isStreaming, verificationStatus, faceStatus])

    // Periodic quality check
    useEffect(() => {
        if (!isStreaming || verificationStatus === 'success' || verificationStatus === 'verifying') return

        const interval = setInterval(checkQuality, 800)
        return () => clearInterval(interval)
    }, [isStreaming, verificationStatus, checkQuality])

    const handleVerify = async () => {
        if (!videoRef.current || !canvasRef.current) return

        setVerificationStatus('verifying')
        setErrorMessage(null)
        setAutoVerifyProgress(0)

        const video = videoRef.current
        const canvas = canvasRef.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d').drawImage(video, 0, 0)

        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData()
                formData.append('image', blob, 'face.jpg')

                const response = await fetch(`${API_URL}/auth/verify-face`, {
                    method: 'POST',
                    body: formData
                })

                const result = await response.json()

                if (result.success && result.is_real_face) {
                    setVerifiedStudent({
                        id: result.student_id,
                        name: result.student_name,
                        role: result.role,
                        confidence: result.confidence
                    })
                    setVerificationStatus('success')
                } else {
                    setErrorMessage(result.error_message || 'Không nhận diện được sinh viên')
                    setVerificationStatus('failed')
                }
            } catch (err) {
                setErrorMessage('Lỗi kết nối server')
                setVerificationStatus('failed')
            }
        }, 'image/jpeg', 0.9)
    }

    const handleProceed = () => {
        if (verifiedStudent.role === 'ADMIN') {
            navigate('/admin', { state: { admin: verifiedStudent } })
        } else {
            navigate('/return', { state: { student: verifiedStudent } })
        }
    }

    const handleRetry = () => {
        setVerificationStatus('idle')
        setVerifiedStudent(null)
        setErrorMessage(null)
        setFaceStatus('waiting')
        setAutoVerifyProgress(0)
    }

    return (
        <div className="verification-screen">
            <header className="verify-header">
                <button className="back-btn" onClick={() => navigate('/')}>
                    ← Quay lại
                </button>
                <h2>Xác thực khuôn mặt</h2>
                <div className="spacer"></div>
            </header>

            <main className="verify-content">
                {/* Camera View */}
                <div className={`camera-container ${verificationStatus} ${faceStatus}`}>
                    <video ref={videoRef} autoPlay playsInline muted />
                    <canvas ref={canvasRef} style={{ display: 'none' }} />

                    <div className="face-overlay">
                        <div className={`face-circle ${faceStatus}`}></div>
                    </div>

                    {/* Quality indicator */}
                    {qualityScore !== null && verificationStatus === 'idle' && (
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

                    {/* Auto-verify progress */}
                    {autoVerifyProgress > 0 && autoVerifyProgress < 100 && verificationStatus === 'idle' && (
                        <div className="auto-verify-progress">
                            <div className="progress-ring">
                                <svg viewBox="0 0 100 100">
                                    <circle className="progress-ring-bg" cx="50" cy="50" r="45" />
                                    <circle
                                        className="progress-ring-fill"
                                        cx="50"
                                        cy="50"
                                        r="45"
                                        style={{
                                            strokeDasharray: `${autoVerifyProgress * 2.83} 283`,
                                            transform: 'rotate(-90deg)',
                                            transformOrigin: 'center'
                                        }}
                                    />
                                </svg>
                            </div>
                            <span className="auto-verify-text">Đang xác thực...</span>
                        </div>
                    )}

                    {verificationStatus === 'verifying' && (
                        <div className="verifying-overlay">
                            <div className="spinner-large"></div>
                            <p>Đang xác thực...</p>
                        </div>
                    )}

                    {verificationStatus === 'success' && (
                        <div className="success-overlay">
                            <div className="check-icon">✓</div>
                        </div>
                    )}
                </div>

                {/* Status message */}
                {verificationStatus === 'idle' && (
                    <div className={`status-message ${faceStatus}`}>
                        <span className="status-icon">
                            {faceStatus === 'valid' ? '✓' : faceStatus === 'invalid' ? '!' : '○'}
                        </span>
                        {statusMessage}
                    </div>
                )}

                {/* Result */}
                {verificationStatus === 'success' && verifiedStudent && (
                    <div className="verify-result success animate-fade-in">
                        <div className="result-icon">👋</div>
                        <h3>Xin chào, {verifiedStudent.name}!</h3>
                        <p>Mã SV: {verifiedStudent.id}</p>
                        <p className="confidence">Độ chính xác: {(verifiedStudent.confidence * 100).toFixed(1)}%</p>
                        <button className="btn btn-success btn-large" onClick={handleProceed}>
                            Tiếp tục mượn/trả sách →
                        </button>
                    </div>
                )}

                {verificationStatus === 'failed' && (
                    <div className="verify-result failed animate-fade-in">
                        <div className="result-icon">❌</div>
                        <h3>Không nhận diện được</h3>
                        <p>{errorMessage}</p>
                        <button className="btn btn-secondary btn-large" onClick={handleRetry}>
                            Thử lại
                        </button>
                    </div>
                )}

                {verificationStatus === 'idle' && (
                    <div className="verify-actions animate-fade-in">
                        <p className="auto-hint">🤖 <strong>Tự động xác thực</strong> khi khuôn mặt đủ điều kiện</p>
                        <button
                            className="btn btn-secondary btn-small manual-btn"
                            onClick={handleVerify}
                            disabled={!isStreaming}
                        >
                            🔍 Xác thực thủ công
                        </button>
                    </div>
                )}
            </main>
        </div>
    )
}
