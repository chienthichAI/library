import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_URL } from '../config'
import { FilesetResolver, FaceDetector } from '@mediapipe/tasks-vision'
import './VerificationScreen.css'

export default function VerificationScreen() {
    const navigate = useNavigate()
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const overlayRef = useRef(null)
    const streamRef = useRef(null)
    const smoothedBboxRef = useRef(null) // For smooth tracking animation
    const animationFrameRef = useRef(null) // For HUD rotation
    
    // WebSockets & MediaPipe refs
    const faceDetectorRef = useRef(null)
    const wsRef = useRef(null)
    const trackingLoopRef = useRef(null)
    const lastFrameTimeRef = useRef(0)
    const redirectTimeoutRef = useRef(null)

    const [isStreaming, setIsStreaming] = useState(false)
    const [verificationStatus, _setVerificationStatus] = useState('idle') // idle, checking, verifying, success, failed
    const verificationStatusRef = useRef('idle')

    const setVerificationStatus = (status) => {
        verificationStatusRef.current = status
        _setVerificationStatus(status)
    }

    const [qualityScore, setQualityScore] = useState(null)
    const [faceStatus, setFaceStatus] = useState('waiting') // waiting, valid, invalid
    const [autoVerifyProgress, setAutoVerifyProgress] = useState(0)
    const [statusMessage, setStatusMessage] = useState('Đang khởi động camera...')
    const [detectedFaces, setDetectedFaces] = useState([]) // Bbox data for UI tracking
    const [verifiedStudent, setVerifiedStudent] = useState(null) // { id, name, role, confidence }
    const [errorMessage, setErrorMessage] = useState(null)

    useEffect(() => {
        initFaceDetector().then(() => {
            connectWebSocket()
            startCamera()
        })
        return () => {
            stopCamera()
            if (redirectTimeoutRef.current) clearTimeout(redirectTimeoutRef.current)
            if (wsRef.current) wsRef.current.close()
            if (trackingLoopRef.current) cancelAnimationFrame(trackingLoopRef.current)
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
        
        ws.onclose = (event) => {
            // code 1000 = server closed cleanly after success — do nothing, UI handles it
            if (event.code !== 1000) {
                console.warn('WebSocket closed unexpectedly, code:', event.code)
                // Will be reconnected by handleRetry if user retries
            }
        }

        ws.onerror = (err) => {
            console.error('WebSocket error:', err)
        }
        
        ws.onmessage = (event) => {
            // Ignore late packets once flow already reached terminal state.
            const currentStatus = verificationStatusRef.current
            if (currentStatus === 'failed' || currentStatus === 'success') {
                return
            }

            let result
            try {
                result = JSON.parse(event.data)
            } catch (err) {
                console.warn('Invalid WebSocket payload:', err)
                return
            }
            
            if (result.status === "failed") {
                const isFaceMissingError = result.error_message?.toLowerCase().includes('khuôn mặt')
                if ((result.quality_issues && result.quality_issues.length > 0) || isFaceMissingError) {
                    setFaceStatus('invalid')
                    setStatusMessage(result.error_message || "Vui lòng xem lại vị trí khuôn mặt")
                    setQualityScore(result.quality_score || 0)
                } else {
                    setErrorMessage(result.error_message || 'Xác thực không thành công')
                    setVerificationStatus('failed')
                    setFaceStatus('invalid')
                    if (trackingLoopRef.current) {
                        cancelAnimationFrame(trackingLoopRef.current)
                        trackingLoopRef.current = null
                    }
                    // Hard-stop websocket so backend cannot continue pushing new auth results.
                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                        wsRef.current.close(1000, 'terminal_failed')
                    }
                }
            } else if (result.status === "success") {
                const student = {
                    id: result.student_id,
                    name: result.student_name,
                    role: result.role,
                    confidence: result.confidence,
                    verification_token: result.verification_token
                }
                setVerifiedStudent(student)
                setVerificationStatus('success')
                setFaceStatus('valid')
                setStatusMessage("Xác thực thành công!")
                if (trackingLoopRef.current) {
                    cancelAnimationFrame(trackingLoopRef.current)
                    trackingLoopRef.current = null
                }
                // Auto-navigate sau 3 giây
                redirectTimeoutRef.current = setTimeout(() => {
                    if (student.role === 'ADMIN') {
                        navigate('/admin', { state: { admin: student } })
                    } else {
                        navigate('/return', { state: { student } })
                    }
                }, 3000)
            } else if (result.status === "no_face_detected" || result.status === "no_prominent_face") {
                setFaceStatus('waiting')
                setStatusMessage(result.error_message || "Đang tìm khuôn mặt...")
                setQualityScore(null)
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
                    setStatusMessage('Sẵn sàng xác thực. Vui lòng nhìn thẳng.')
                    trackingLoopRef.current = requestAnimationFrame(trackingLoop)
                }
            }
        } catch (err) {
            setErrorMessage('Không thể truy cập camera. Vui lòng kiểm tra quyền!')
            setStatusMessage('Lỗi thiết bị')
        }
    }

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
        }
        if (videoRef.current) videoRef.current.srcObject = null
    }

    const drawFaceBoxes = useCallback((detections) => {
        const overlay = overlayRef.current
        const video = videoRef.current
        const canvas = canvasRef.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d').drawImage(video, 0, 0)

        // Only checking visually in background but don't block UI strictly unless necessary
        // setFaceStatus('checking')

        canvas.toBlob(async (blob) => {
            if (!blob) return;
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
                        setStatusMessage('Đã khóa mục tiêu. Giữ im vị trí!')

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
                        setStatusMessage(result.message || 'Vui lòng căn chỉnh lại khuôn mặt')

                        // Cancel auto-verify if quality drops
                        if (autoVerifyTimerRef.current) {
                            clearTimeout(autoVerifyTimerRef.current)
                            autoVerifyTimerRef.current = null
                            setAutoVerifyProgress(0)
                        }
                    }
                } else {
                    setFaceStatus('invalid')
                    setStatusMessage('Ánh sáng yếu hoặc không rõ mặt')

                    if (autoVerifyTimerRef.current) {
                        clearTimeout(autoVerifyTimerRef.current)
                        autoVerifyTimerRef.current = null
                        setAutoVerifyProgress(0)
                    }
                }
            } catch (err) {
                console.log('Quality API error:', err)
                setFaceStatus('waiting')
            }
        }, 'image/jpeg', 0.8)
    }, [isStreaming, verificationStatus, faceStatus])

    // Periodic quality check
    useEffect(() => {
        if (!isStreaming || verificationStatus !== 'idle') return

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

        // Scale down to 640x360 to drastically speed up toBlob encoding and network upload 
        // 15 frames of 1280x720 at 0.95 quality is heavy!
        canvas.width = 640
        canvas.height = 360

        setStatusMessage('Đang lấy mẫu sinh trắc học...')

        const frames = []
        const requiredFrames = 15 // 15 frames ~ 0.5s for rPPG + screen flicker

        const captureInterval = setInterval(() => {
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
            canvas.toBlob((blob) => {
                if (blob) {
                    frames.push(blob)
                }

                if (frames.length >= requiredFrames) {
                    clearInterval(captureInterval)
                    sendForAuth(frames)
                }
            }, 'image/jpeg', 0.8)
        }, 33)

        const sendForAuth = async (capturedFrames) => {
            try {
                const formData = new FormData()
                capturedFrames.forEach((frame, index) => {
                    formData.append('images', frame, `face_${index}.jpg`)
                })

                setStatusMessage('Mã hóa & Xác thực mã định danh...')
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
                    setDetectedFaces([])
                    smoothedBboxRef.current = null // Reset smoothing
                    const ctx = overlayRef.current?.getContext('2d')
                    if (ctx) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
                }
            }
            // If status is NOT idle (e.g. error, success), we keep the last frame or clear it?
            // On failure, we should probably clear the overlay eventually or let handleRetry do it.
        }
        
        // Stop the loop on terminal states — do not waste CPU after authentication is done
        const currentStatus = verificationStatusRef.current
        if (currentStatus === 'success' || currentStatus === 'failed') {
            trackingLoopRef.current = null
            return
        }
        
        trackingLoopRef.current = requestAnimationFrame(trackingLoop)
    }, [drawFaceBoxes]) // Removed verificationStatus as dependency - use ref instead

    // Since verification is now continuous, we don't have a separate handleVerify method
    // But we override it incase the UI button wants to do something
    const handleVerify = () => {}

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

        // Clear and Reset overlay
        const ctx = overlayRef.current?.getContext('2d')
        if (ctx) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
        smoothedBboxRef.current = null

        // Reconnect WebSocket if closed
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            connectWebSocket()
        }

        // Restart loop if stopped
        if (trackingLoopRef.current) {
            cancelAnimationFrame(trackingLoopRef.current)
            trackingLoopRef.current = null
        }
        trackingLoopRef.current = requestAnimationFrame(trackingLoop)
    }

    // Helper for circular progress
    const radius = 64;
    const circumference = 2 * Math.PI * radius;
    // autoVerifyProgress is obsolete now, but we keep it here to avoid deleting UI logic blindly
    const strokeDashoffset = circumference - (0 / 100) * circumference;

    return (
        <div className="verification-screen">
            <header className="verify-header">
                <button className="back-btn" onClick={() => navigate('/')}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '8px', verticalAlign: 'middle' }}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Quay lại
                </button>
                <h2>Xác Thực Sinh Trắc Học</h2>
                <div style={{ width: '120px' }}></div>
            </header>

            <main className="verify-content">

                <div style={{ display: (verificationStatus === 'idle' || verificationStatus === 'verifying') ? 'block' : 'none', width: '100%' }}>
                    <div className="setup-container">

                        {/* Status Message Display */}
                        {verificationStatus === 'idle' && (
                            <div className={`status-pill ${faceStatus}`}>
                                <div className="status-icon-animated">
                                    {faceStatus === 'valid' ? (
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                        </svg>
                                    ) : faceStatus === 'invalid' ? (
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                        </svg>
                                    ) : (
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <circle cx="12" cy="12" r="10" />
                                            <circle cx="12" cy="12" r="3" fill="currentColor" />
                                        </svg>
                                    )}
                                </div>
                                <span>{statusMessage}</span>
                            </div>
                        )}

                        <div className={`camera-wrapper ${faceStatus}`}>
                            <div className={`camera-container ${verificationStatus}`}>
                                <video ref={videoRef} autoPlay playsInline muted />
                                <canvas ref={overlayRef} className="face-tracking-overlay" />
                                <canvas ref={canvasRef} style={{ display: 'none' }} />

                                <div className={`tracking-badge ${detectedFaces.length > 0 ? 'active' : ''}`}>
                                    <span className="dot"></span>
                                    {detectedFaces.length > 0 ? 'Đang Theo Dõi' : 'Đang Tìm...'}
                                </div>

                                <div className="face-overlay">
                                    {/* Removed face-circle guide as requested */}
                                </div>

                                {/* Quality indicator */}
                                {qualityScore !== null && verificationStatus === 'idle' && (
                                    <div className={`quality-indicator ${faceStatus}`}>
                                        <div className="quality-header">
                                            <span>Chất lượng nhận diện</span>
                                            <span>{Math.round(qualityScore * 100)}%</span>
                                        </div>
                                        <div className="quality-bar">
                                            <div
                                                className="quality-fill"
                                                style={{ width: `${qualityScore * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Auto-verify progress */}
                                {autoVerifyProgress > 0 && autoVerifyProgress < 100 && verificationStatus === 'idle' && (
                                    <div className="auto-verify-progress">
                                        <div className="progress-ring">
                                            <svg viewBox="0 0 140 140">
                                                <circle className="progress-ring-bg" cx="70" cy="70" r={radius} />
                                                <circle
                                                    className="progress-ring-fill"
                                                    cx="70"
                                                    cy="70"
                                                    r={radius}
                                                    style={{ strokeDasharray: circumference, strokeDashoffset: strokeDashoffset }}
                                                />
                                            </svg>
                                            <span style={{ position: 'absolute', fontWeight: 'bold', fontSize: '1.2rem', color: '#fff' }}>
                                                {Math.round(autoVerifyProgress)}%
                                            </span>
                                        </div>
                                        <span className="auto-verify-text-premium">Chuẩn bị xác thực...</span>
                                    </div>
                                )}

                                {verificationStatus === 'verifying' && (
                                    <div className="verifying-overlay">
                                        <div className="spinner-premium"></div>
                                        <p>Đang trích xuất đặc trưng AI...</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {verificationStatus === 'idle' && (
                            <div className="verify-actions animate-fade-in">
                                <div className="auto-verify-banner">
                                    <div className="auto-verify-pulse"></div>
                                    Tự động nhận diện khi khuôn mặt rõ nét
                                </div>
                                <button
                                    className="manual-btn"
                                    onClick={handleVerify}
                                    disabled={!isStreaming || faceStatus !== 'valid'}
                                    style={{ opacity: (!isStreaming || faceStatus !== 'valid') ? 0.4 : 1 }}
                                >
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                                        <circle cx="12" cy="12" r="3" />
                                    </svg>
                                    Quét Thủ Công
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* Result Cards */}
                {verificationStatus === 'success' && verifiedStudent && (
                    <div className="verify-result success animate-fade-in">
                        <div className="result-avatar-container">
                            <div className="result-avatar">
                                🎓
                            </div>
                        </div>
                        <h3>Xin chào, {verifiedStudent.name}!</h3>
                        <div className="student-id">Mã số sinh viên: <strong>{verifiedStudent.id}</strong></div>
                        <p className="confidence">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Độ tương khớp: {(verifiedStudent.confidence * 100).toFixed(1)}%
                        </p>

                        <div style={{ marginTop: '32px' }}>
                            <button className="btn btn-success btn-large" onClick={handleProceed} style={{ width: '100%', borderRadius: '100px', fontSize: '1.2rem', padding: '20px', boxShadow: '0 8px 24px rgba(34, 197, 94, 0.4)' }}>
                                {verifiedStudent.role === 'ADMIN' ? 'Truy Cập Quản Trị Hệ Thống' : 'Tiếp Tục Mượn/Trả Sách'}
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginLeft: '12px' }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                </svg>
                            </button>
                        </div>
                    </div>
                )}

                {verificationStatus === 'failed' && (
                    <div className="verify-result failed animate-fade-in">
                        <div className="result-avatar-container" style={{ borderColor: 'rgba(239, 68, 68, 0.5)' }}>
                            <div className="result-avatar" style={{ background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}>
                                🚫
                            </div>
                        </div>
                        <h3>Truy Cập Bị Từ Chối</h3>
                        <p style={{ color: '#f87171', fontSize: '1.1rem', margin: '16px 0 32px' }}>{errorMessage}</p>

                        <button className="btn btn-secondary btn-large" onClick={handleRetry} style={{ width: '100%', borderRadius: '100px', fontSize: '1.2rem', padding: '20px' }}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '12px' }}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                            Thử Lại Ngay
                        </button>
                    </div>
                )}
            </main>
        </div>
    )
}

