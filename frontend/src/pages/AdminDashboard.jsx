import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import FaceCapture from '../components/FaceCapture'
import { API_URL } from '../config'
import './AdminDashboard.css'
import AIChatbot from '../components/AIChatbot'

// ─── Constants ───────────────────────────────────────────────────────────────
const PAGE_SIZE = 20

const SCAN_STEPS = [
    { id: 'front', label: 'Bìa trước', icon: '📖', desc: 'Lấy tên sách & tác giả' },
    { id: 'back',  label: 'Bìa sau',   icon: '📊', desc: 'Lấy mã vạch / Barcode' },
    { id: 'info',  label: 'Trang đầu', icon: '📝', desc: 'Bổ sung thông tin' }
]

const EMPTY_BOOK = { book_id: '', title: '', author: '', barcode: '', subject_category: '', description: '' }

// ─── Status badge helper ──────────────────────────────────────────────────────
const statusLabel = {
    AVAILABLE: 'Sẵn sàng', BORROWED: 'Đang mượn',
    DAMAGED: 'Hỏng', LOST: 'Mất', RESERVED: 'Đặt trước',
    ACTIVE: 'Hoạt động', SUSPENDED: 'Đình chỉ', GRADUATED: 'Tốt nghiệp', INACTIVE: 'Không HĐ'
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function AdminDashboard() {
    const navigate   = useNavigate()
    const location   = useLocation()
    const admin      = location.state?.admin
    const authHeaders = admin?.verification_token
        ? { Authorization: `Bearer ${admin.verification_token}` }
        : {}

    // Tabs
    const [activeTab, setActiveTab] = useState('stats')

    // Stats
    const [stats, setStats] = useState({ books: 0, students: 0, active_loans: 0, overdue: 0, fine_pending: 0 })

    // Books state
    const [books,          setBooks]          = useState([])
    const [booksTotal,     setBooksTotal]     = useState(0)
    const [booksOffset,    setBooksOffset]    = useState(0)
    const [bookSearch,     setBookSearch]     = useState('')
    const [bookStatusFilter, setBookStatusFilter] = useState('')
    const [editingBook,    setEditingBook]    = useState(null)  // {book_id, ...} or null
    const [isAddingBook,   setIsAddingBook]   = useState(false)

    // Students state
    const [students,        setStudents]        = useState([])
    const [studentsTotal,   setStudentsTotal]   = useState(0)
    const [studentsOffset,  setStudentsOffset]  = useState(0)
    const [studentSearch,   setStudentSearch]   = useState('')
    const [studentStatusFilter, setStudentStatusFilter] = useState('')
    const [editingStudent,  setEditingStudent]  = useState(null)
    const [registerFaceStudent, setRegisterFaceStudent] = useState(null)

    // General
    const [isLoading,  setIsLoading]  = useState(false)
    const [isScanning, setIsScanning] = useState(false)
    const [scannedShots, setScannedShots] = useState([])
    const [newBook, setNewBook] = useState(EMPTY_BOOK)

    // Camera
    const videoRef  = useRef(null)
    const streamRef = useRef(null)
    const canvasRef = useRef(null)

    const currentStepIndex = Math.min(scannedShots.length, SCAN_STEPS.length - 1)

    // ── Auth guard ────────────────────────────────────────────────────────────
    useEffect(() => {
        if (!admin || admin.role !== 'ADMIN') { navigate('/verify'); return }
        fetchStats()
    }, [admin])

    // ── Tab change ────────────────────────────────────────────────────────────
    useEffect(() => {
        if (activeTab === 'books')    { setBooksOffset(0);    fetchBooks(0) }
        if (activeTab === 'students') { setStudentsOffset(0); fetchStudents(0) }
    }, [activeTab])

    // ── Camera lifecycle ──────────────────────────────────────────────────────
    useEffect(() => {
        if (isAddingBook) startCamera()
        else stopCamera()
        return () => stopCamera()
    }, [isAddingBook])

    // ─── Camera helpers ───────────────────────────────────────────────────────
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
            console.error('Camera error:', err)
        }
    }

    const stopCamera = () => {
        streamRef.current?.getTracks().forEach(t => t.stop())
        streamRef.current = null
    }

    // ─── Book scanning ────────────────────────────────────────────────────────
    const scanBookInfo = async () => {
        if (!videoRef.current || isScanning) return
        setIsScanning(true)
        try {
            const canvas = canvasRef.current
            const video  = videoRef.current
            canvas.width  = video.videoWidth
            canvas.height = video.videoHeight
            canvas.getContext('2d').drawImage(video, 0, 0)

            const imageUrl = canvas.toDataURL('image/jpeg')
            const blob     = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85))
            const fd       = new FormData()
            fd.append('image', blob, 'scan.jpg')

            const res    = await fetch(`${API_URL}/books/detect`, { method: 'POST', body: fd })
            if (!res.ok) throw new Error('API Error')
            const result = await res.json()

            // Merge scan result into form (don't overwrite existing good data)
            setNewBook(prev => ({
                book_id:          result.book_id   || result.barcode || prev.book_id,
                title:            result.title      || prev.title,
                author:           result.author     || prev.author,
                barcode:          result.barcode    || prev.barcode,
                subject_category: result.subject_category || prev.subject_category,
                description:      result.description      || prev.description
            }))

            setScannedShots(prev => [...prev, { type: SCAN_STEPS[currentStepIndex].label, url: imageUrl, data: result }])
        } catch (err) {
            console.error('Scan failed:', err)
            alert('Không thể kết nối Backend.')
        } finally {
            setIsScanning(false)
        }
    }

    const resetScan = () => { setScannedShots([]); setNewBook(EMPTY_BOOK) }

    // ─── Stats ────────────────────────────────────────────────────────────────
    const fetchStats = async () => {
        try {
            const [booksRes, studentsRes, overdueRes] = await Promise.all([
                fetch(`${API_URL}/books/?limit=1`),
                fetch(`${API_URL}/students/?limit=1`),
                fetch(`${API_URL}/transactions/stats/overdue`)
            ])

            const booksData    = booksRes.ok    ? await booksRes.json()    : {}
            const studentsData = studentsRes.ok ? await studentsRes.json() : {}
            const overdueData  = overdueRes.ok  ? await overdueRes.json()  : {}

            // Count borrowed via separate call (small overhead, accurate count)
            const borrowedRes  = booksRes.ok
                ? await fetch(`${API_URL}/books/?status=BORROWED&limit=1`)
                : null
            const borrowedData = borrowedRes?.ok ? await borrowedRes.json() : {}

            setStats({
                books:        booksData.total        ?? 0,
                students:     studentsData.total     ?? 0,
                active_loans: borrowedData.total     ?? 0,
                overdue:      overdueData.total_overdue    ?? 0,
                fine_pending: overdueData.total_fine_pending ?? 0
            })
        } catch (err) {
            console.error('Fetch stats error:', err)
        }
    }

    // ─── Books CRUD ───────────────────────────────────────────────────────────
    const fetchBooks = useCallback(async (offset = booksOffset) => {
        setIsLoading(true)
        try {
            const params = new URLSearchParams({ limit: PAGE_SIZE, offset })
            if (bookSearch)        params.set('search', bookSearch)
            if (bookStatusFilter)  params.set('status', bookStatusFilter)
            const res  = await fetch(`${API_URL}/books/?${params}`)
            if (!res.ok) throw new Error()
            const data = await res.json()
            setBooks(data.books ?? [])
            setBooksTotal(data.total ?? 0)
            setBooksOffset(offset)
        } catch { setBooks([]) }
        finally { setIsLoading(false) }
    }, [bookSearch, bookStatusFilter, booksOffset])

    const handleAddBook = async (e) => {
        e.preventDefault()
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/books/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...authHeaders },
                body: JSON.stringify(newBook)
            })
            if (res.ok) {
                alert('✅ Thêm sách thành công!')
                resetScan(); setIsAddingBook(false); fetchBooks(0); fetchStats()
            } else {
                const err = await res.json()
                alert(`❌ Lỗi: ${err.detail || 'Không thể thêm sách'}`)
            }
        } catch { alert('Lỗi kết nối server') }
        finally { setIsLoading(false) }
    }

    const handleUpdateBook = async (e) => {
        e.preventDefault()
        if (!editingBook) return
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/books/${editingBook.book_id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', ...authHeaders },
                body: JSON.stringify({
                    title:  editingBook.title,
                    author: editingBook.author,
                    status: editingBook.status
                })
            })
            if (res.ok) {
                alert('✅ Cập nhật thành công!')
                setEditingBook(null); fetchBooks(booksOffset); fetchStats()
            } else {
                const err = await res.json()
                alert(`❌ ${err.detail}`)
            }
        } catch { alert('Lỗi kết nối server') }
        finally { setIsLoading(false) }
    }

    const handleDeleteBook = async (book) => {
        if (!window.confirm(`Xóa sách "${book.title}"? Thao tác không thể hoàn tác.`)) return
        try {
            const res = await fetch(`${API_URL}/books/${book.book_id}`, {
                method: 'DELETE',
                headers: authHeaders
            })
            if (res.status === 204) {
                alert('✅ Đã xóa sách!')
                fetchBooks(booksOffset); fetchStats()
            } else {
                const err = await res.json()
                alert(`❌ ${err.detail}`)
            }
        } catch { alert('Lỗi kết nối server') }
    }

    // ─── Students CRUD ────────────────────────────────────────────────────────
    const fetchStudents = useCallback(async (offset = studentsOffset) => {
        setIsLoading(true)
        try {
            const params = new URLSearchParams({ limit: PAGE_SIZE, offset })
            if (studentSearch)        params.set('search', studentSearch)
            if (studentStatusFilter)  params.set('status', studentStatusFilter)
            const res  = await fetch(`${API_URL}/students/?${params}`)
            if (!res.ok) throw new Error()
            const data = await res.json()
            setStudents(data.students ?? [])
            setStudentsTotal(data.total ?? 0)
            setStudentsOffset(offset)
        } catch { setStudents([]) }
        finally { setIsLoading(false) }
    }, [studentSearch, studentStatusFilter, studentsOffset])

    const handleUpdateStudent = async (e) => {
        e.preventDefault()
        if (!editingStudent) return
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/students/${editingStudent.student_id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', ...authHeaders },
                body: JSON.stringify({
                    full_name: editingStudent.full_name,
                    email:     editingStudent.email,
                    phone:     editingStudent.phone,
                    status:    editingStudent.status
                })
            })
            if (res.ok) {
                alert('✅ Cập nhật sinh viên thành công!')
                setEditingStudent(null); fetchStudents(studentsOffset)
            } else {
                const err = await res.json()
                alert(`❌ ${err.detail}`)
            }
        } catch { alert('Lỗi kết nối server') }
        finally { setIsLoading(false) }
    }

    const handleDeleteStudent = async (student) => {
        if (!window.confirm(`Xóa sinh viên "${student.full_name}"?`)) return
        try {
            const res = await fetch(`${API_URL}/students/${student.student_id}`, {
                method: 'DELETE',
                headers: authHeaders
            })
            if (res.status === 204) {
                alert('✅ Đã xóa sinh viên!')
                fetchStudents(studentsOffset)
            } else {
                const err = await res.json()
                alert(`❌ ${err.detail}`)
            }
        } catch { alert('Lỗi kết nối server') }
    }

    const handleClearFine = async (student) => {
        if (!window.confirm(`Xóa ${student.fine_balance.toLocaleString()} VND tiền phạt của ${student.full_name}?`)) return
        try {
            const res = await fetch(`${API_URL}/students/${student.student_id}/clear-fine`, {
                method: 'POST',
                headers: authHeaders
            })
            if (res.ok) {
                alert('✅ Đã xóa tiền phạt!')
                fetchStudents(studentsOffset)
            } else {
                const err = await res.json()
                alert(`❌ ${err.detail}`)
            }
        } catch { alert('Lỗi kết nối server') }
    }

    const handleFaceCaptureDone = async (images) => {
        setIsLoading(true)
        try {
            let registeredCount = 0;
            let lastError = null;

            for (let i = 0; i < images.length; i++) {
                const formData = new FormData()
                formData.append('student_id', registerFaceStudent.student_id)
                formData.append('image', images[i].blob)

                try {
                    const faceResponse = await fetch(`${API_URL}/auth/register-face`, {
                        method: 'POST',
                        body: formData
                    })

                    if (!faceResponse.ok) {
                        const err = await faceResponse.json()
                        lastError = err.detail || 'Không thể đăng ký một góc khuôn mặt'
                        console.warn(`Face registration failed for image ${i + 1}:`, lastError)
                    } else {
                        registeredCount++;
                    }
                } catch (e) {
                    lastError = e.message;
                    console.warn(`Face registration fetch failed for image ${i + 1}:`, e)
                }
            }

            if (registeredCount === 0) {
                alert(`❌ Đăng ký khuôn mặt thất bại: ${lastError}`)
            } else {
                alert(`✅ Đăng ký thành công ${registeredCount} góc mặt cho sinh viên ${registerFaceStudent.full_name}!`)
            }
        } finally {
            setIsLoading(false)
            setRegisterFaceStudent(null)
            fetchStudents(studentsOffset)
        }
    }

    // ─── Pagination helper ────────────────────────────────────────────────────
    const Pagination = ({ total, offset, onPage }) => {
        const totalPages = Math.ceil(total / PAGE_SIZE)
        const currentPage = Math.floor(offset / PAGE_SIZE)
        if (totalPages <= 1) return null
        return (
            <div className="pagination">
                <button disabled={currentPage === 0} onClick={() => onPage((currentPage - 1) * PAGE_SIZE)}>‹ Trước</button>
                <span>Trang {currentPage + 1} / {totalPages} &nbsp;·&nbsp; {total} kết quả</span>
                <button disabled={currentPage >= totalPages - 1} onClick={() => onPage((currentPage + 1) * PAGE_SIZE)}>Sau ›</button>
            </div>
        )
    }

    // ─── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="admin-dashboard">
            {/* Sidebar */}
            <aside className="admin-sidebar">
                <div className="admin-profile">
                    <div className="admin-avatar">A</div>
                    <div className="admin-info">
                        <h3>{admin?.name || admin?.full_name}</h3>
                        <span className="badge-admin">Administrator</span>
                    </div>
                </div>

                <nav className="admin-nav">
                    {[
                        { id: 'stats',    icon: '📊', label: 'Tổng quan' },
                        { id: 'books',    icon: '📚', label: 'Quản lý sách' },
                        { id: 'students', icon: '👥', label: 'Quản lý sinh viên' },
                        { id: 'chatbot',  icon: '🤖', label: 'Trợ lý AI' }
                    ].map(tab => (
                        <button key={tab.id}
                            className={activeTab === tab.id ? 'active' : ''}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            {tab.icon} {tab.label}
                        </button>
                    ))}

                    <div className="nav-divider" />

                    <button className="test-kiosk-btn" onClick={() => navigate('/return', { state: { student: admin } })}>
                        🤖 Chế độ Kiosk
                    </button>
                </nav>

                <div className="sidebar-footer">
                    <button className="logout-btn" onClick={() => navigate('/')}>🚪 Đăng xuất</button>
                </div>
            </aside>

            {/* Main */}
            <main className="admin-main">
                {activeTab !== 'chatbot' && (
                    <header className="admin-header">
                        <h2>
                            {activeTab === 'stats' ? 'Tổng quan hệ thống'
                                : activeTab === 'books' ? 'Thư viện sách'
                                : activeTab === 'students' ? 'Danh sách sinh viên'
                                : ''}
                        </h2>
                    </header>
                )}

                <section className={`admin-content ${activeTab === 'chatbot' ? 'chatbot-full' : ''}`}>

                    {/* ── Stats tab ── */}
                    {activeTab === 'stats' && (
                        <div className="stats-grid">
                            {[
                                { icon: '📚', value: stats.books,        label: 'Tổng số sách' },
                                { icon: '👥', value: stats.students,     label: 'Sinh viên đăng ký' },
                                { icon: '🔄', value: stats.active_loans, label: 'Sách đang mượn' },
                                { icon: '⚠️', value: stats.overdue,      label: 'Quá hạn' },
                                { icon: '💰', value: `${(stats.fine_pending / 1000).toFixed(0)}k`, label: 'Tiền phạt tồn đọng (VND)' }
                            ].map(card => (
                                <div className="stat-card" key={card.label}>
                                    <span className="stat-icon">{card.icon}</span>
                                    <div className="stat-data">
                                        <span className="stat-value">{card.value}</span>
                                        <span className="stat-label">{card.label}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* ── Books tab ── */}
                    {activeTab === 'books' && (
                        <div className="books-management">
                            {/* Toolbar */}
                            <div className="table-toolbar">
                                <input
                                    className="search-input"
                                    placeholder="🔍 Tìm theo tên, tác giả, mã sách..."
                                    value={bookSearch}
                                    onChange={e => setBookSearch(e.target.value)}
                                    onKeyDown={e => e.key === 'Enter' && fetchBooks(0)}
                                />
                                <select value={bookStatusFilter} onChange={e => { setBookStatusFilter(e.target.value); fetchBooks(0) }}>
                                    <option value="">— Trạng thái —</option>
                                    <option value="AVAILABLE">Sẵn sàng</option>
                                    <option value="BORROWED">Đang mượn</option>
                                    <option value="DAMAGED">Hỏng</option>
                                    <option value="LOST">Mất</option>
                                </select>
                                <button className="btn btn-ghost" onClick={() => fetchBooks(0)}>Lọc</button>
                                <button className="btn btn-primary" onClick={() => setIsAddingBook(true)}>➕ Thêm sách</button>
                            </div>

                            {/* Table */}
                            <div className="admin-table-container">
                                {isLoading ? <div className="loading-row">⏳ Đang tải...</div> : (
                                    <table className="admin-table">
                                        <thead>
                                            <tr>
                                                <th>Mã Sách</th>
                                                <th>Tiêu đề</th>
                                                <th>Tác giả</th>
                                                <th>Barcode</th>
                                                <th>Trạng thái</th>
                                                <th>Thao tác</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {books.length === 0 ? (
                                                <tr><td colSpan={6} style={{ textAlign: 'center', padding: '2rem' }}>Không có dữ liệu</td></tr>
                                            ) : books.map(book => (
                                                <tr key={book.book_id}>
                                                    <td>{book.book_id}</td>
                                                    <td><strong>{book.title}</strong></td>
                                                    <td>{book.author}</td>
                                                    <td><code>{book.barcode}</code></td>
                                                    <td>
                                                        <span className={`status-pill ${book.status.toLowerCase()}`}>
                                                            {statusLabel[book.status] ?? book.status}
                                                        </span>
                                                    </td>
                                                    <td className="action-cell">
                                                        <button className="btn-icon" title="Sửa" onClick={() => setEditingBook({ ...book })}>✏️</button>
                                                        <button className="btn-icon danger" title="Xóa" onClick={() => handleDeleteBook(book)}
                                                            disabled={book.status === 'BORROWED'}>🗑️</button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </div>
                            <Pagination total={booksTotal} offset={booksOffset} onPage={off => fetchBooks(off)} />
                        </div>
                    )}

                    {/* ── Students tab ── */}
                    {activeTab === 'students' && (
                        <div className="students-management">
                            {/* Toolbar */}
                            <div className="table-toolbar">
                                <input
                                    className="search-input"
                                    placeholder="🔍 Tìm theo tên hoặc mã sinh viên..."
                                    value={studentSearch}
                                    onChange={e => setStudentSearch(e.target.value)}
                                    onKeyDown={e => e.key === 'Enter' && fetchStudents(0)}
                                />
                                <select value={studentStatusFilter} onChange={e => { setStudentStatusFilter(e.target.value); fetchStudents(0) }}>
                                    <option value="">— Trạng thái —</option>
                                    <option value="ACTIVE">Hoạt động</option>
                                    <option value="SUSPENDED">Đình chỉ</option>
                                    <option value="GRADUATED">Tốt nghiệp</option>
                                    <option value="INACTIVE">Không HĐ</option>
                                </select>
                                <button className="btn btn-ghost" onClick={() => fetchStudents(0)}>Lọc</button>
                            </div>

                            {/* Table */}
                            <div className="admin-table-container">
                                {isLoading ? <div className="loading-row">⏳ Đang tải...</div> : (
                                    <table className="admin-table">
                                        <thead>
                                            <tr>
                                                <th>Mã SV</th>
                                                <th>Họ tên</th>
                                                <th>Email</th>
                                                <th>Tiền phạt</th>
                                                <th>Trạng thái</th>
                                                <th>Thao tác</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {students.length === 0 ? (
                                                <tr><td colSpan={6} style={{ textAlign: 'center', padding: '2rem' }}>Không có dữ liệu</td></tr>
                                            ) : students.map(student => (
                                                <tr key={student.student_id}>
                                                    <td>{student.student_id}</td>
                                                    <td><strong>{student.full_name}</strong></td>
                                                    <td>{student.email}</td>
                                                    <td className={student.fine_balance > 0 ? 'text-danger' : ''}>
                                                        {student.fine_balance.toLocaleString()} VND
                                                    </td>
                                                    <td>
                                                        <span className={`status-pill ${student.status.toLowerCase()}`}>
                                                            {statusLabel[student.status] ?? student.status}
                                                        </span>
                                                    </td>
                                                    <td className="action-cell">
                                                        <button className="btn-icon" title="Cập nhật khuôn mặt" onClick={() => setRegisterFaceStudent(student)}>📸</button>
                                                        <button className="btn-icon" title="Sửa" onClick={() => setEditingStudent({ ...student })}>✏️</button>
                                                        {student.fine_balance > 0 && (
                                                            <button className="btn-icon warning" title="Xóa phạt" onClick={() => handleClearFine(student)}>💸</button>
                                                        )}
                                                        <button className="btn-icon danger" title="Xóa" onClick={() => handleDeleteStudent(student)}>🗑️</button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </div>
                            <Pagination total={studentsTotal} offset={studentsOffset} onPage={off => fetchStudents(off)} />
                        </div>
                    )}

                    {/* ── Chatbot tab ── */}
                    {activeTab === 'chatbot' && (
                        <div className="chatbot-tab-content">
                            <AIChatbot fullPage={true} />
                        </div>
                    )}

                </section>
            </main>

            {/* ════════════════════════════════════════════
                Modal: Add Book
            ════════════════════════════════════════════ */}
            {isAddingBook && (
                <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setIsAddingBook(false)}>
                    <div className="modal-content animate-slide-up">
                        <div className="modal-header">
                            <h3>Khai báo sách mới</h3>
                            <button className="close-modal" onClick={() => setIsAddingBook(false)}>&times;</button>
                        </div>

                        {/* Scanner section */}
                        <div className="scanner-section">
                            <div className="scanner-workflow">
                                {SCAN_STEPS.map((step, idx) => (
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
                                <div className="scanner-overlay"><div className="scan-region" /></div>
                                <button
                                    type="button"
                                    className={`btn-scan ${isScanning ? 'scanning' : ''}`}
                                    onClick={scanBookInfo}
                                    disabled={isScanning || scannedShots.length >= SCAN_STEPS.length}
                                >
                                    {isScanning ? '🔄 Đang nhận diện...' : `📸 Chụp ${SCAN_STEPS[currentStepIndex].label}`}
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

                        {/* Book form */}
                        <form onSubmit={handleAddBook}>
                            {[
                                { label: 'Mã ID Sách (Duy nhất)', key: 'book_id', placeholder: 'Ví dụ: MS001', required: true },
                                { label: 'Tiêu đề sách', key: 'title', placeholder: 'Tên cuốn sách', required: true },
                                { label: 'Tác giả', key: 'author', placeholder: 'Tên tác giả', required: true },
                                { label: 'Mã Barcode (nhận diện camera)', key: 'barcode', placeholder: 'Số barcode', required: true },
                                { label: 'Thể loại', key: 'subject_category', placeholder: 'Công nghệ, Ngoại ngữ...' },
                            ].map(field => (
                                <div className="form-group" key={field.key}>
                                    <label>{field.label}</label>
                                    <input
                                        type="text"
                                        placeholder={field.placeholder}
                                        value={newBook[field.key]}
                                        onChange={e => setNewBook(prev => ({ ...prev, [field.key]: e.target.value }))}
                                        required={field.required}
                                    />
                                </div>
                            ))}
                            <div className="form-group">
                                <label>Mô tả chi tiết</label>
                                <textarea
                                    placeholder="Nội dung tóm tắt của sách..."
                                    value={newBook.description}
                                    onChange={e => setNewBook(prev => ({ ...prev, description: e.target.value }))}
                                    rows={3}
                                />
                            </div>
                            <div className="modal-actions">
                                <button type="button" className="btn btn-secondary" onClick={() => setIsAddingBook(false)}>Hủy</button>
                                <button type="submit" className="btn btn-primary" disabled={isLoading}>
                                    {isLoading ? 'Đang lưu...' : '💾 Lưu thông tin'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* ════════════════════════════════════════════
                Modal: Edit Book
            ════════════════════════════════════════════ */}
            {editingBook && (
                <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setEditingBook(null)}>
                    <div className="modal-content animate-slide-up" style={{ maxWidth: 480 }}>
                        <div className="modal-header">
                            <h3>Sửa thông tin sách</h3>
                            <button className="close-modal" onClick={() => setEditingBook(null)}>&times;</button>
                        </div>
                        <form onSubmit={handleUpdateBook}>
                            <div className="form-group">
                                <label>Tiêu đề</label>
                                <input type="text" value={editingBook.title}
                                    onChange={e => setEditingBook(prev => ({ ...prev, title: e.target.value }))} required />
                            </div>
                            <div className="form-group">
                                <label>Tác giả</label>
                                <input type="text" value={editingBook.author || ''}
                                    onChange={e => setEditingBook(prev => ({ ...prev, author: e.target.value }))} />
                            </div>
                            <div className="form-group">
                                <label>Trạng thái</label>
                                <select value={editingBook.status}
                                    onChange={e => setEditingBook(prev => ({ ...prev, status: e.target.value }))}
                                    disabled={editingBook.status === 'BORROWED'}>
                                    <option value="AVAILABLE">Sẵn sàng</option>
                                    <option value="DAMAGED">Hỏng</option>
                                    <option value="LOST">Mất</option>
                                    {editingBook.status === 'BORROWED' && <option value="BORROWED">Đang mượn (không thể đổi)</option>}
                                </select>
                                {editingBook.status === 'BORROWED' && (
                                    <small style={{ color: '#f59e0b' }}>⚠️ Sách đang mượn — trạng thái chỉ thay đổi sau khi trả</small>
                                )}
                            </div>
                            <div className="modal-actions">
                                <button type="button" className="btn btn-secondary" onClick={() => setEditingBook(null)}>Hủy</button>
                                <button type="submit" className="btn btn-primary" disabled={isLoading}>
                                    {isLoading ? 'Đang lưu...' : '💾 Lưu'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* ════════════════════════════════════════════
                Modal: Edit Student
            ════════════════════════════════════════════ */}
            {editingStudent && (
                <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setEditingStudent(null)}>
                    <div className="modal-content animate-slide-up" style={{ maxWidth: 480 }}>
                        <div className="modal-header">
                            <h3>Sửa thông tin sinh viên</h3>
                            <button className="close-modal" onClick={() => setEditingStudent(null)}>&times;</button>
                        </div>
                        <form onSubmit={handleUpdateStudent}>
                            <div className="form-group">
                                <label>Họ tên</label>
                                <input type="text" value={editingStudent.full_name}
                                    onChange={e => setEditingStudent(prev => ({ ...prev, full_name: e.target.value }))} required />
                            </div>
                            <div className="form-group">
                                <label>Email</label>
                                <input type="email" value={editingStudent.email || ''}
                                    onChange={e => setEditingStudent(prev => ({ ...prev, email: e.target.value }))} />
                            </div>
                            <div className="form-group">
                                <label>Số điện thoại</label>
                                <input type="tel" value={editingStudent.phone || ''}
                                    onChange={e => setEditingStudent(prev => ({ ...prev, phone: e.target.value }))} />
                            </div>
                            <div className="form-group">
                                <label>Trạng thái tài khoản</label>
                                <select value={editingStudent.status}
                                    onChange={e => setEditingStudent(prev => ({ ...prev, status: e.target.value }))}>
                                    <option value="ACTIVE">Hoạt động</option>
                                    <option value="SUSPENDED">Đình chỉ</option>
                                    <option value="GRADUATED">Tốt nghiệp</option>
                                    <option value="INACTIVE">Không hoạt động</option>
                                </select>
                            </div>
                            <div className="modal-actions">
                                <button type="button" className="btn btn-secondary" onClick={() => setEditingStudent(null)}>Hủy</button>
                                <button type="submit" className="btn btn-primary" disabled={isLoading}>
                                    {isLoading ? 'Đang lưu...' : '💾 Lưu'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* ════════════════════════════════════════════
                Modal: Register Face
            ════════════════════════════════════════════ */}
            {registerFaceStudent && (
                <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setRegisterFaceStudent(null)}>
                    <div className="modal-content animate-slide-up" style={{ maxWidth: 800 }}>
                        <div className="modal-header">
                            <h3>Chụp khuôn mặt - {registerFaceStudent.full_name} ({registerFaceStudent.student_id})</h3>
                            <button className="close-modal" onClick={() => setRegisterFaceStudent(null)}>&times;</button>
                        </div>
                        <div className="modal-body">
                            <FaceCapture
                                onCapture={handleFaceCaptureDone}
                                requiredCaptures={3}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

