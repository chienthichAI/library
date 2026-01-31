import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { API_URL } from '../config'
import './BookReturnScreen.css'

export default function BookReturnScreen() {
    const navigate = useNavigate()
    const location = useLocation()
    const student = location.state?.student

    const [isStreaming, setIsStreaming] = useState(false)
    const [detectionStatus, setDetectionStatus] = useState('idle') // idle, detecting, found, not_found
    const [isProcessing, setIsProcessing] = useState(false)
    const [borrowingInfo, setBorrowingInfo] = useState(null)
    const [detectedBook, setDetectedBook] = useState(null)
    const [transactionResult, setTransactionResult] = useState(null)

    // Camera states
    const [devices, setDevices] = useState([])
    const [faceDeviceId, setFaceDeviceId] = useState('')
    const [bookDeviceId, setBookDeviceId] = useState('')
    const [isFaceStreaming, setIsFaceStreaming] = useState(false)

    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const faceVideoRef = useRef(null)
    const bookStreamRef = useRef(null)
    const faceStreamRef = useRef(null)

    useEffect(() => {
        if (!student) {
            navigate('/verify')
            return
        }
        // Small delay to ensure previous component released the camera
        const timer = setTimeout(() => {
            initCameras()
        }, 300)

        fetchBorrowingInfo()
        return () => {
            clearTimeout(timer)
            stopAllCameras()
        }
    }, [student, navigate])

    const initCameras = async () => {
        try {
            const allDevices = await navigator.mediaDevices.enumerateDevices()
            const videoDevices = allDevices.filter(d => d.kind === 'videoinput')
            setDevices(videoDevices)

            if (videoDevices.length >= 2) {
                let bookStream = null;
                let faceStream = null;
                try {
                    // Try to start both cameras independently
                    console.log("Attempting dual camera setup...")
                    bookStream = await navigator.mediaDevices.getUserMedia({
                        video: { deviceId: { exact: videoDevices[1].deviceId }, width: 640, height: 480 }
                    })
                    faceStream = await navigator.mediaDevices.getUserMedia({
                        video: { deviceId: { exact: videoDevices[0].deviceId }, width: 640, height: 480 }
                    })

                    bookStreamRef.current = bookStream
                    faceStreamRef.current = faceStream

                    if (videoRef.current) {
                        videoRef.current.srcObject = bookStream
                        videoRef.current.play().catch(e => console.error("Book video play error:", e))
                        setIsStreaming(true)
                    }
                    if (faceVideoRef.current) {
                        faceVideoRef.current.srcObject = faceStream
                        faceVideoRef.current.play().catch(e => console.error("Face video play error:", e))
                        setIsFaceStreaming(true)
                    }
                    setBookDeviceId(videoDevices[1].deviceId)
                    setFaceDeviceId(videoDevices[0].deviceId)
                } catch (e) {
                    console.warn("Dual camera setup failed, falling back to single camera mode:", e)
                    // If dual setup fails, release any tracks that might have opened
                    if (bookStream) bookStream.getTracks().forEach(t => t.stop())
                    if (faceStream) faceStream.getTracks().forEach(t => t.stop())

                    await setupSingleCamera(videoDevices[0].deviceId)
                }
            } else if (videoDevices.length > 0) {
                await setupSingleCamera(videoDevices[0].deviceId)
            }
        } catch (err) {
            console.error('Error listing cameras:', err)
        }
    }

    const setupSingleCamera = async (deviceId) => {
        try {
            console.log("Setting up single camera mode for device:", deviceId)
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 640, height: 480 }
            })

            bookStreamRef.current = stream
            faceStreamRef.current = stream

            if (videoRef.current) {
                videoRef.current.srcObject = stream
                videoRef.current.play().catch(e => console.error("Book video play error:", e))
                setIsStreaming(true)
            }
            if (faceVideoRef.current) {
                faceVideoRef.current.srcObject = stream
                faceVideoRef.current.play().catch(e => console.error("Face video play error:", e))
                setIsFaceStreaming(true)
            }
            setBookDeviceId(deviceId || 'default')
            setFaceDeviceId(deviceId || 'default')
        } catch (e) {
            console.error("Single camera access error:", e)
        }
    }

    const fetchBorrowingInfo = async () => {
        try {
            const response = await fetch(`${API_URL}/students/${student.id}/borrowing-info`)
            const data = await response.json()
            setBorrowingInfo(data)
        } catch (err) {
            console.error('Error fetching borrowing info:', err)
        }
    }



    const stopAllCameras = () => {
        [bookStreamRef, faceStreamRef].forEach(ref => {
            if (ref.current) {
                ref.current.getTracks().forEach(track => track.stop())
                ref.current = null
            }
        });

        if (videoRef.current) videoRef.current.srcObject = null;
        if (faceVideoRef.current) faceVideoRef.current.srcObject = null;
    }

    const handleDetectBook = async () => {
        if (!videoRef.current || !canvasRef.current) return

        setDetectionStatus('detecting')

        const video = videoRef.current
        const canvas = canvasRef.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d').drawImage(video, 0, 0)

        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData()
                formData.append('image', blob, 'book.jpg')

                const response = await fetch(`${API_URL}/books/detect`, {
                    method: 'POST',
                    body: formData
                })

                const result = await response.json()

                if (result.success && result.book_exists) {
                    setDetectedBook({
                        id: result.book_id,
                        title: result.title,
                        author: result.author,
                        barcode: result.barcode,
                        status: result.status,
                        confidence: result.detection_confidence
                    })
                    setDetectionStatus('found')
                } else {
                    setDetectedBook(null)
                    setDetectionStatus('not_found')
                }
            } catch (err) {
                console.error('Detection error:', err)
                setDetectionStatus('not_found')
            }
        }, 'image/jpeg', 0.9)
    }

    const handleBorrow = async () => {
        if (!detectedBook || !student) return
        setIsProcessing(true)

        try {
            const response = await fetch(`${API_URL}/transactions/borrow`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    student_id: student.id,
                    book_id: detectedBook.id
                })
            })

            const result = await response.json()
            if (result.success) {
                fetchBorrowingInfo()
            }
            setTransactionResult({
                type: 'borrow',
                success: result.success,
                book_title: result.book_title,
                due_date: result.due_date,
                error: result.error_message
            })
        } catch (err) {
            setTransactionResult({ type: 'borrow', success: false, error: 'Lỗi kết nối' })
        } finally {
            setIsProcessing(false)
        }
    }

    const handleReturn = async () => {
        if (!detectedBook || !student) return

        setIsProcessing(true)

        try {
            // Step 1: Server-side validation to avoid race conditions
            const validateResponse = await fetch(`${API_URL}/transactions/validate-return`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    student_id: student.id,
                    book_id: detectedBook.id
                })
            })

            const validateResult = await validateResponse.json()

            if (!validateResult.can_return) {
                setTransactionResult({
                    type: 'return',
                    success: false,
                    error: validateResult.error_message || 'Bạn chưa mượn cuốn sách này. Vui lòng kiểm tra lại.'
                })
                setIsProcessing(false)
                return
            }

            // Step 2: Process the return
            const response = await fetch(`${API_URL}/transactions/return`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    student_id: student.id,
                    book_id: detectedBook.id
                })
            })

            const result = await response.json()
            if (result.success) {
                fetchBorrowingInfo()
            }
            setTransactionResult({
                type: 'return',
                success: result.success,
                book_title: result.book_title,
                days_overdue: result.days_overdue,
                fine_amount: result.fine_amount,
                error: result.error_message
            })
        } catch (err) {
            setTransactionResult({ type: 'return', success: false, error: 'Lỗi kết nối' })
        } finally {
            setIsProcessing(false)
        }
    }

    const handleReset = () => {
        setDetectionStatus('idle')
        setDetectedBook(null)
        setTransactionResult(null)
    }

    if (!student) return null

    return (
        <div className="book-return-screen">
            <header className="book-header">
                <button className="back-btn" onClick={() => navigate('/')}>
                    ← Trang chủ
                </button>
                <div className="student-info-bar">
                    <div className="security-cam-mini">
                        <video ref={faceVideoRef} autoPlay playsInline muted />
                        <div className="rec-dot"></div>
                    </div>
                    <span>👤 {student.name}</span>
                    <span className="muted">({student.id})</span>
                    {borrowingInfo && (
                        <div className="borrowing-stats">
                            <span className="stat-badge">📚 {borrowingInfo.currently_borrowed}/{borrowingInfo.max_books} cuốn</span>
                            {borrowingInfo.fine_balance > 0 && (
                                <span className="stat-badge danger">💰 {borrowingInfo.fine_balance.toLocaleString('vi-VN')} VND</span>
                            )}
                        </div>
                    )}
                </div>
                <div className="spacer"></div>
            </header>

            <main className="book-content-layout">
                {/* Left Side: Active Borrowings */}
                <aside className="active-borrowings-panel">
                    <h3>📚 Sách đang mượn</h3>
                    {!borrowingInfo ? (
                        <div className="loading-small">Đang tải...</div>
                    ) : borrowingInfo.borrowed_books.length === 0 ? (
                        <p className="empty-message">Chưa mượn cuốn sách nào</p>
                    ) : (
                        <div className="borrowed-books-list">
                            {borrowingInfo.borrowed_books.map(book => (
                                <div key={book.transaction_id} className={`borrowed-book-card ${book.is_overdue ? 'overdue' : ''}`}>
                                    <div className="book-card-header">
                                        <span className="book-title">{book.title}</span>
                                        {book.is_overdue && <span className="overdue-tag">Trễ hạn</span>}
                                    </div>
                                    <div className="book-card-details">
                                        <span>📅 Mượn: {new Date(book.borrow_date).toLocaleDateString('vi-VN')}</span>
                                        <span>⏳ Còn: {book.days_left} ngày</span>
                                    </div>
                                    {book.fine_amount > 0 && (
                                        <div className="book-card-fine">Phạt: {book.fine_amount.toLocaleString('vi-VN')} VND</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </aside>

                {/* Right Side: Main Actions */}
                <div className="book-main-actions">
                    {!transactionResult ? (
                        <>
                            {/* Camera for book detection */}
                            <div className={`book-camera ${detectionStatus}`}>
                                <video ref={videoRef} autoPlay playsInline muted />
                                <canvas ref={canvasRef} style={{ display: 'none' }} />

                                <div className="book-guide">
                                    <div className="guide-box"></div>
                                    <div className="guide-hint">
                                        <span>💡 Cần đủ ánh sáng</span>
                                        <span>📏 Cách camera 15-20cm</span>
                                    </div>
                                    <span className="main-guide-text">Đặt mã vạch hoặc bìa sách vào khung</span>
                                </div>

                                {detectionStatus === 'detecting' && (
                                    <div className="detecting-overlay">
                                        <div className="spinner-large"></div>
                                        <p>Đang nhận diện sách...</p>
                                    </div>
                                )}
                            </div>

                            {/* Detection result */}
                            {detectionStatus === 'found' && detectedBook && (
                                <div className="book-result animate-fade-in">
                                    <div className="book-info-card">
                                        <div className="book-icon">📖</div>
                                        <div className="book-details">
                                            <h3>{detectedBook.title}</h3>
                                            <p>{detectedBook.author}</p>
                                            <p className="barcode">Mã: {detectedBook.barcode}</p>
                                            <p className={`status ${detectedBook.status.toLowerCase()}`}>
                                                Hệ thống nhận diện: <strong>{detectedBook.status === 'AVAILABLE' ? 'MƯỢN SÁCH' : 'TRẢ SÁCH'}</strong>
                                            </p>
                                        </div>
                                    </div>

                                    <div className="action-buttons">
                                        {detectedBook.status === 'AVAILABLE' ? (
                                            <button
                                                className="btn btn-success btn-large"
                                                onClick={handleBorrow}
                                                disabled={isProcessing}
                                            >
                                                {isProcessing ? 'Đang xử lý...' : '📥 Mượn sách'}
                                            </button>
                                        ) : (
                                            <button
                                                className="btn btn-primary btn-large"
                                                onClick={handleReturn}
                                                disabled={isProcessing}
                                            >
                                                {isProcessing ? 'Đang xử lý...' : '📤 Trả sách'}
                                            </button>
                                        )}
                                        <button className="btn btn-secondary" onClick={handleReset}>
                                            Quét sách khác
                                        </button>
                                    </div>
                                </div>
                            )}

                            {detectionStatus === 'not_found' && (
                                <div className="book-result not-found animate-fade-in">
                                    <p>❌ Không nhận diện được sách. Vui lòng thử lại.</p>
                                    <button className="btn btn-secondary" onClick={handleReset}>
                                        Thử lại
                                    </button>
                                </div>
                            )}

                            {detectionStatus === 'idle' && (
                                <div className="book-actions animate-fade-in">
                                    <p>Đặt sách lên bàn và nhấn nút quét</p>
                                    <button
                                        className="btn btn-primary btn-large"
                                        onClick={handleDetectBook}
                                        disabled={!isStreaming}
                                    >
                                        📷 Quét sách
                                    </button>
                                </div>
                            )}
                        </>
                    ) : (
                        /* Transaction result */
                        <div className="transaction-result animate-fade-in">
                            {transactionResult.success ? (
                                <div className="result-success">
                                    <div className="success-icon">✓</div>
                                    <h2>{transactionResult.type === 'borrow' ? 'Mượn sách thành công!' : 'Trả sách thành công!'}</h2>
                                    <p className="book-title">📚 {transactionResult.book_title}</p>

                                    {transactionResult.type === 'borrow' && (
                                        <div className="due-info">
                                            <span>📅 Hạn trả:</span>
                                            <strong>{transactionResult.due_date}</strong>
                                        </div>
                                    )}

                                    {transactionResult.type === 'return' && transactionResult.fine_amount > 0 && (
                                        <div className="fine-info">
                                            <span>⚠️ Trễ {transactionResult.days_overdue} ngày</span>
                                            <strong>Phạt: {transactionResult.fine_amount.toLocaleString('vi-VN')} VND</strong>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="result-error">
                                    <div className="error-icon">✕</div>
                                    <h2>Thất bại</h2>
                                    <p>{transactionResult.error}</p>
                                </div>
                            )}

                            <div className="result-actions">
                                <button className="btn btn-primary btn-large" onClick={handleReset}>
                                    Mượn/Trả sách khác
                                </button>
                                <button className="btn btn-secondary" onClick={() => navigate('/')}>
                                    Về trang chủ
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div>
    )
}
