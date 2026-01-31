import { useState, useEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { API_URL } from '../config'
import './AdminDashboard.css'

export default function AdminDashboard() {
    const navigate = useNavigate()
    const location = useLocation()
    const admin = location.state?.admin

    const [activeTab, setActiveTab] = useState('stats')
    const [stats, setStats] = useState({ books: 0, students: 0, active_loans: 0 })
    const [books, setBooks] = useState([])
    const [students, setStudents] = useState([])
    const [isAddingBook, setIsAddingBook] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [isScanning, setIsScanning] = useState(false)
    const [scannedShots, setScannedShots] = useState([]) // Array of { type, url, data }
    const videoRef = useRef(null)
    const streamRef = useRef(null)
    const canvasRef = useRef(null)

    const scanSteps = [
        { id: 'front', label: 'Bìa trước', icon: '📖', desc: 'Lấy tên sách & tác giả' },
        { id: 'back', label: 'Bìa sau', icon: '📊', desc: 'Lấy mã vạch / Barcode' },
        { id: 'info', label: 'Trang đầu', icon: '📝', desc: 'Bổ sung thông tin' }
    ]
    const currentStepIndex = Math.min(scannedShots.length, scanSteps.length - 1)

    // Form state for new book
    const [newBook, setNewBook] = useState({
        book_id: '',
        title: '',
        author: '',
        barcode: ''
    })

    useEffect(() => {
        if (!admin || admin.role !== 'ADMIN') {
            navigate('/verify')
            return
        }
        fetchStats()
    }, [admin])

    useEffect(() => {
        if (activeTab === 'books') fetchBooks()
        if (activeTab === 'students') fetchStudents()
    }, [activeTab])

    // Handle camera for scanning
    useEffect(() => {
        if (isAddingBook) {
            startCamera()
        } else {
            stopCamera()
        }
        return () => stopCamera()
    }, [isAddingBook])

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: 640, height: 480 }
            })
            if (videoRef.current) {
                videoRef.current.srcObject = stream
                streamRef.current = stream
            }
        } catch (err) {
            console.error('Cant start camera for scanning:', err)
        }
    }

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
        }
    }

    const scanBookInfo = async () => {
        if (!videoRef.current || isScanning) return

        setIsScanning(true)
        try {
            const canvas = canvasRef.current
            const video = videoRef.current
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            canvas.getContext('2d').drawImage(video, 0, 0)

            const imageUrl = canvas.toDataURL('image/jpeg')
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8))
            const formData = new FormData()
            formData.append('image', blob, 'scan.jpg')

            const res = await fetch(`${API_URL}/books/detect`, {
                method: 'POST',
                body: formData
            })

            if (!res.ok) throw new Error('API Error')
            const result = await res.json()

            // Update form with NEW findings (don't overwrite if empty)
            setNewBook(prev => ({
                book_id: result.book_id || result.barcode || prev.book_id,
                title: result.title || prev.title,
                author: result.author || prev.author,
                barcode: result.barcode || prev.barcode
            }))

            // Save this shot
            setScannedShots(prev => [...prev, {
                type: scanSteps[currentStepIndex].label,
                url: imageUrl,
                data: result
            }])

        } catch (err) {
            console.error('Scan failed:', err)
            alert('Không thể kết nối Backend. Hãy chắc chắn server đang chạy.')
        } finally {
            setIsScanning(false)
        }
    }

    const resetScan = () => {
        setScannedShots([])
        setNewBook({ book_id: '', title: '', author: '', barcode: '' })
    }

    const fetchStats = async () => {
        try {
            // In a real app, you'd have a stats endpoint. 
            // For now, let's fetch books and students and count.
            const [booksRes, studentsRes] = await Promise.all([
                fetch(`${API_URL}/books/`),
                fetch(`${API_URL}/students/`)
            ])
            const booksData = await booksRes.json()
            const studentsData = await studentsRes.json()

            setStats({
                books: booksData.length,
                students: studentsData.length,
                active_loans: booksData.filter(b => b.status === 'BORROWED').length
            })
        } catch (err) {
            console.error('Error fetching stats:', err)
        }
    }

    const fetchBooks = async () => {
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/books/`)
            const data = await res.json()
            setBooks(data)
        } catch (err) {
            console.error('Error fetching books:', err)
        } finally {
            setIsLoading(false)
        }
    }

    const fetchStudents = async () => {
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/students/`)
            const data = await res.json()
            setStudents(data)
        } catch (err) {
            console.error('Error fetching students:', err)
        } finally {
            setIsLoading(false)
        }
    }

    const handleAddBook = async (e) => {
        e.preventDefault()
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/books/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newBook)
            })
            if (res.ok) {
                alert('Thêm sách thành công!')
                resetScan()
                setIsAddingBook(false)
                fetchBooks()
                fetchStats()
            } else {
                const err = await res.json()
                alert(`Lỗi: ${err.detail || 'Không thể thêm sách'}`)
            }
        } catch (err) {
            alert('Lỗi kết nối server')
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="admin-dashboard">
            <aside className="admin-sidebar">
                <div className="admin-profile">
                    <div className="admin-avatar">A</div>
                    <div className="admin-info">
                        <h3>{admin?.name}</h3>
                        <span className="badge-admin">Administrator</span>
                    </div>
                </div>

                <nav className="admin-nav">
                    <button
                        className={activeTab === 'stats' ? 'active' : ''}
                        onClick={() => setActiveTab('stats')}
                    >
                        📊 Tổng quan
                    </button>
                    <button
                        className={activeTab === 'books' ? 'active' : ''}
                        onClick={() => setActiveTab('books')}
                    >
                        📚 Quản lý sách
                    </button>
                    <button
                        className={activeTab === 'students' ? 'active' : ''}
                        onClick={() => setActiveTab('students')}
                    >
                        👥 Quản lý sinh viên
                    </button>

                    <div className="nav-divider"></div>

                    <button
                        className="test-kiosk-btn"
                        onClick={() => navigate('/return', { state: { student: admin } })}
                    >
                        🤖 Chế độ Kiosk (Mượn/Trả)
                    </button>
                </nav>

                <div className="sidebar-footer">
                    <button className="logout-btn" onClick={() => navigate('/')}>
                        🚪 Đăng xuất
                    </button>
                </div>
            </aside>

            <main className="admin-main">
                <header className="admin-header">
                    <h2>{activeTab === 'stats' ? 'Tổng quan hệ thống' : activeTab === 'books' ? 'Thư viện sách' : 'Danh sách sinh viên'}</h2>
                </header>

                <section className="admin-content">
                    {activeTab === 'stats' && (
                        <div className="stats-grid">
                            <div className="stat-card">
                                <span className="stat-icon">📚</span>
                                <div className="stat-data">
                                    <span className="stat-value">{stats.books}</span>
                                    <span className="stat-label">Tổng số sách</span>
                                </div>
                            </div>
                            <div className="stat-card">
                                <span className="stat-icon">👥</span>
                                <div className="stat-data">
                                    <span className="stat-value">{stats.students}</span>
                                    <span className="stat-label">Sinh viên đăng ký</span>
                                </div>
                            </div>
                            <div className="stat-card">
                                <span className="stat-icon">🔄</span>
                                <div className="stat-data">
                                    <span className="stat-value">{stats.active_loans}</span>
                                    <span className="stat-label">Sách đang mượn</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'books' && (
                        <div className="books-management">
                            <div className="table-actions">
                                <button className="btn btn-primary" onClick={() => setIsAddingBook(true)}>
                                    ➕ Khai báo sách mới
                                </button>
                            </div>

                            <div className="admin-table-container">
                                <table className="admin-table">
                                    <thead>
                                        <tr>
                                            <th>Mã Sách</th>
                                            <th>Tiêu đề</th>
                                            <th>Tác giả</th>
                                            <th>Barcode</th>
                                            <th>Trạng thái</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {books.map(book => (
                                            <tr key={book.book_id}>
                                                <td>{book.book_id}</td>
                                                <td><strong>{book.title}</strong></td>
                                                <td>{book.author}</td>
                                                <td><code>{book.barcode}</code></td>
                                                <td>
                                                    <span className={`status-pill ${book.status.toLowerCase()}`}>
                                                        {book.status === 'AVAILABLE' ? 'Sẵn sàng' : 'Đang mượn'}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {activeTab === 'students' && (
                        <div className="students-management">
                            <div className="admin-table-container">
                                <table className="admin-table">
                                    <thead>
                                        <tr>
                                            <th>Mã SV</th>
                                            <th>Họ tên</th>
                                            <th>Email</th>
                                            <th>Số dư phạt</th>
                                            <th>Trạng thái</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {students.map(student => (
                                            <tr key={student.student_id}>
                                                <td>{student.student_id}</td>
                                                <td><strong>{student.full_name}</strong></td>
                                                <td>{student.email}</td>
                                                <td>{student.fine_balance.toLocaleString()} VND</td>
                                                <td>
                                                    <span className={`status-pill ${student.status.toLowerCase()}`}>
                                                        {student.status}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </section>
            </main>

            {/* Add Book Modal */}
            {isAddingBook && (
                <div className="modal-overlay">
                    <div className="modal-content animate-slide-up">
                        <div className="modal-header">
                            <h3>Khai báo sách mới</h3>
                            <button className="close-modal" onClick={() => setIsAddingBook(false)}>&times;</button>
                        </div>

                        <div className="scanner-section">
                            <div className="scanner-workflow">
                                {scanSteps.map((step, idx) => (
                                    <div key={step.id} className={`step-item ${idx === scannedShots.length ? 'active' : ''} ${idx < scannedShots.length ? 'done' : ''}`}>
                                        <span className="step-icon">{idx < scannedShots.length ? '✅' : step.icon}</span>
                                        <div className="step-text">
                                            <label>{step.label}</label>
                                            <small>{step.desc}</small>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <div className="scanner-container">
                                <video ref={videoRef} autoPlay playsInline muted className="scanner-preview" />
                                <canvas ref={canvasRef} style={{ display: 'none' }} />
                                <div className="scanner-overlay">
                                    <div className="scan-region"></div>
                                </div>
                                <button
                                    type="button"
                                    className={`btn-scan ${isScanning ? 'scanning' : ''}`}
                                    onClick={scanBookInfo}
                                    disabled={isScanning || scannedShots.length >= scanSteps.length}
                                >
                                    {isScanning ? '🔄 Đang nhận diện...' : `📸 Chụp ${scanSteps[currentStepIndex].label}`}
                                </button>
                            </div>

                            {scannedShots.length > 0 && (
                                <div className="scanned-gallery">
                                    {scannedShots.map((shot, i) => (
                                        <div key={i} className="gallery-item">
                                            <img src={shot.url} alt="scan" />
                                            <span>{shot.type}</span>
                                        </div>
                                    ))}
                                    <button type="button" className="btn-reset-scan" onClick={resetScan}>🔄 Làm mới</button>
                                </div>
                            )}
                        </div>

                        <form onSubmit={handleAddBook}>
                            <div className="form-group">
                                <label>Mã ID Sách (Duy nhất)</label>
                                <input
                                    type="text"
                                    placeholder="Ví dụ: MS001"
                                    value={newBook.book_id}
                                    onChange={e => setNewBook({ ...newBook, book_id: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Tiêu đề sách</label>
                                <input
                                    type="text"
                                    placeholder="Tên cuốn sách"
                                    value={newBook.title}
                                    onChange={e => setNewBook({ ...newBook, title: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Tác giả</label>
                                <input
                                    type="text"
                                    placeholder="Tên tác giả"
                                    value={newBook.author}
                                    onChange={e => setNewBook({ ...newBook, author: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Mã Barcode (Để nhận diện)</label>
                                <input
                                    type="text"
                                    placeholder="Số barcode hoặc mã định danh"
                                    value={newBook.barcode}
                                    onChange={e => setNewBook({ ...newBook, barcode: e.target.value })}
                                    required
                                />
                                <small>Đây là mã mà hệ thống dùng camera để khớp sách.</small>
                            </div>
                            <div className="modal-actions">
                                <button type="button" className="btn btn-secondary" onClick={() => setIsAddingBook(false)}>Hủy</button>
                                <button type="submit" className="btn btn-primary" disabled={isLoading}>
                                    {isLoading ? 'Đang lưu...' : 'Lưu thông tin'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    )
}
