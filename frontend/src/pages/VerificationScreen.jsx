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
                
                // Store student info for Global Chatbot access
                sessionStorage.setItem('smartlib_student_id', student.id);
                sessionStorage.setItem('smartlib_student_name', student.name);
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
        if (!overlay || !video) return

        // Sync overlay size with video display size
        const displayWidth = video.clientWidth
        const displayHeight = video.clientHeight
        
        if (displayWidth === 0 || displayHeight === 0) return

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
            if (!isPrimary) return // Only draw premium HUD for primary face

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
                const lerp = 0.25 // Smoothness factor
                smoothedBboxRef.current.x += (targetX1 - smoothedBboxRef.current.x) * lerp
                smoothedBboxRef.current.y += (targetY1 - smoothedBboxRef.current.y) * lerp
                smoothedBboxRef.current.w += (targetW - smoothedBboxRef.current.w) * lerp
                smoothedBboxRef.current.h += (targetH - smoothedBboxRef.current.h) * lerp
            }

            const { x, y, w, h } = smoothedBboxRef.current
            const x2 = x + w
            const y2 = y + h

            const color = '#22c55e'
            const glowColor = 'rgba(34,197,94,0.4)'
            const now = Date.now() / 1000

            const centerX = x + w/2
            const centerY = y + h/2
            
            // --- DRAW HUD BRACKETS ---
            ctx.save()
            
            // Subtle Background fill
            ctx.fillStyle = 'rgba(34, 197, 94, 0.05)'
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

            // Side Ticks (Optional premium touch)
            ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(x - 5, centerY); ctx.lineTo(x + 5, centerY); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(x2 - 5, centerY); ctx.lineTo(x2 + 5, centerY); ctx.stroke()

            ctx.restore()

            // --- DATA READOUTS ---
            const fontSize = 11
            ctx.font = `${fontSize}px "JetBrains Mono", monospace`
            ctx.fillStyle = color
            
            // ID Badge
            const idText = `SUBJECT_ID: ${faceStatus === 'valid' ? 'AUTHORIZED' : 'SCANNING'}`
            ctx.fillText(idText, x, y - 15)

            // Tech Brackets (Left Data)
            const metrics = [
                `ACC: ${Math.round(confidence * 100)}%`,
                `POS: ${Math.round(x)},${Math.round(y)}`,
                `DIM: ${Math.round(w)}x${Math.round(h)}`
            ]
            
            metrics.forEach((m, i) => {
                ctx.fillStyle = 'rgba(34,197,94,0.7)'
                ctx.fillText(m, x2 + 10, y + (i * (fontSize + 5)) + 15)
            })

            // Scanning Line inside the box (Matrix Effect)
            const scanPos = (Math.sin(now * 2.5) + 1) / 2
            const scanY = y + (h * scanPos)
            const grad = ctx.createLinearGradient(x, scanY, x2, scanY)
            grad.addColorStop(0, 'transparent')
            grad.addColorStop(0.5, 'rgba(34,197,94,0.4)')
            grad.addColorStop(1, 'transparent')
            ctx.fillStyle = grad
            ctx.fillRect(x, scanY - 1, w, 2)
        })
    }, [faceStatus])

    const trackingLoop = useCallback(() => {
        const now = performance.now()
        const status = verificationStatusRef.current
        
        // Sync overlay size with display size before doing anything
        const overlay = overlayRef.current
        const video = videoRef.current
        if (overlay && video && video.readyState >= 2) {
            const displayWidth = video.clientWidth
            const displayHeight = video.clientHeight
            if (displayWidth !== 0 && displayHeight !== 0) {
                if (overlay.width !== displayWidth || overlay.height !== displayHeight) {
                    overlay.width = displayWidth
                    overlay.height = displayHeight
                }
            }
        }

        // Always run detector if camera is ready, even if WS is down
        if (videoRef.current && videoRef.current.readyState >= 2 && faceDetectorRef.current) {
            // ONLY detect and draw if status is idle
            if (status === 'idle') {
                const detections = faceDetectorRef.current.detectForVideo(videoRef.current, now)
                
                if (detections.detections.length > 0) {
                    setDetectedFaces(detections.detections)
                    drawFaceBoxes(detections.detections)
                    
                    // Only send to WS if open and rate limited
                    if (wsRef.current?.readyState === WebSocket.OPEN && now - lastFrameTimeRef.current >= 100) {
                        lastFrameTimeRef.current = now
                        
                        const canvas = canvasRef.current
                        canvas.width = 640
                        canvas.height = 360
                        const ctx = canvas.getContext('2d')
                        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
                        
                        canvas.toBlob((blob) => {
                            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
                                wsRef.current.send(blob)
                            }
                        }, 'image/jpeg', 0.8)
                    }
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
