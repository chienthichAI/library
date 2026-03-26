import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { API_URL } from '../config'
import './StudentDashboard.css'
import AIChatbot from '../components/AIChatbot'

export default function StudentDashboard() {
    const navigate = useNavigate()
    const location = useLocation()
    const student = location.state?.student

    const [activeTab, setActiveTab] = useState('overview')
    const [borrowingInfo, setBorrowingInfo] = useState(null)
    const [books, setBooks] = useState([])
    const [bookSearch, setBookSearch] = useState('')
    const [history, setHistory] = useState([])
    const [isLoading, setIsLoading] = useState(false)

    useEffect(() => {
        if (!student) {
            navigate('/verify')
            return
        }
        fetchBorrowingInfo()
    }, [student, navigate])

    useEffect(() => {
        if (activeTab === 'search') fetchBooks()
        if (activeTab === 'history') fetchHistory()
    }, [activeTab])

    const fetchBorrowingInfo = async () => {
        try {
            const response = await fetch(`${API_URL}/students/${student.id}/borrowing-info`)
            const data = await response.json()
            setBorrowingInfo(data)
        } catch (err) {
            console.error('Error fetching borrowing info:', err)
        }
    }

    const fetchBooks = async () => {
        setIsLoading(true)
        try {
            const params = new URLSearchParams({ limit: 20 })
            if (bookSearch) params.set('search', bookSearch)
            const res = await fetch(`${API_URL}/books/?${params}`)
            const data = await res.json()
            setBooks(data.books || [])
        } catch (err) {
            console.error('Error fetching books:', err)
        } finally {
            setIsLoading(false)
        }
    }

    const fetchHistory = async () => {
        setIsLoading(true)
        try {
            const res = await fetch(`${API_URL}/transactions/history/${student.id}`)
            if (!res.ok) {
                console.error('Failed to fetch history:', res.statusText)
                setHistory([])
                return
            }
            const data = await res.json()
            setHistory(data.transactions || [])
        } catch (err) {
            console.error('Error fetching history:', err)
            setHistory([])
        } finally {
            setIsLoading(false)
        }
    }

    const handleLogout = () => {
        navigate('/')
    }

    if (!student) return null

    return (
        <div className="student-dashboard">
            <aside className="student-sidebar">
                <div className="student-profile">
                    <div className="student-avatar">{student.name?.charAt(0) || 'S'}</div>
                    <div className="student-info">
                        <h3>{student.name}</h3>
                        <span className="student-id">ID: {student.id}</span>
                        <span className="badge-student">Sinh viên</span>
                    </div>
                </div>

                <nav className="student-nav">
                    <button 
                        className={activeTab === 'overview' ? 'active' : ''} 
                        onClick={() => setActiveTab('overview')}
                    >
                        🏠 Tổng quan
                    </button>
                    <button 
                        className={activeTab === 'search' ? 'active' : ''} 
                        onClick={() => setActiveTab('search')}
                    >
                        🔍 Tìm sách
                    </button>
                    <button 
                        className={activeTab === 'history' ? 'active' : ''} 
                        onClick={() => setActiveTab('history')}
                    >
                        📜 Lịch sử mượn
                    </button>
                    
                    <button 
                        className={activeTab === 'chatbot' ? 'active' : ''} 
                        onClick={() => setActiveTab('chatbot')}
                    >
                        🤖 Trợ lý AI
                    </button>
                    
                    <div className="nav-divider" />
                    
                    <button className="kiosk-mode-btn" onClick={() => navigate('/return', { state: { student } })}>
                        🔄 Mượn & Trả sách
                    </button>
                </nav>

                <div className="sidebar-footer">
                    <button className="logout-btn" onClick={handleLogout}>🚪 Đăng xuất</button>
                </div>
            </aside>

            <main className="student-main">
                {activeTab !== 'chatbot' && (
                    <header className="student-header">
                        <h2>
                            {activeTab === 'overview' ? 'Chào mừng quay trở lại!' : 
                             activeTab === 'search' ? 'Thư viện sách' : 
                             activeTab === 'history' ? 'Lịch sử hoạt động' : ''}
                        </h2>
                    </header>
                )}

                <section className={`student-content ${activeTab === 'chatbot' ? 'chatbot-full' : ''}`}>
                    {activeTab === 'overview' && (
                        <div className="overview-tab">
                            <div className="stats-grid">
                                <div className="stat-card">
                                    <span className="stat-icon">📚</span>
                                    <div className="stat-data">
                                        <span className="stat-value">{borrowingInfo?.currently_borrowed || 0}</span>
                                        <span className="stat-label">Sách đang mượn</span>
                                    </div>
                                </div>
                                <div className="stat-card">
                                    <span className="stat-icon">⚠️</span>
                                    <div className="stat-data">
                                        <span className="stat-value">
                                            {borrowingInfo?.borrowed_books?.filter(b => b.is_overdue).length || 0}
                                        </span>
                                        <span className="stat-label">Sách quá hạn</span>
                                    </div>
                                </div>
                                <div className="stat-card">
                                    <span className="stat-icon">💰</span>
                                    <div className="stat-data">
                                        <span className="stat-value">{(borrowingInfo?.fine_balance || 0).toLocaleString()}</span>
                                        <span className="stat-label">Tiền phạt (VND)</span>
                                    </div>
                                </div>
                            </div>

                            <div className="current-borrowings">
                                <h3>📖 Sách đang mượn</h3>
                                {!borrowingInfo ? (
                                    <div className="loading">Đang tải...</div>
                                ) : borrowingInfo.borrowed_books.length === 0 ? (
                                    <div className="empty-state">
                                        <p>Bạn đang không mượn cuốn sách nào.</p>
                                        <button className="btn btn-primary" onClick={() => setActiveTab('search')}>Khám phá thư viện</button>
                                    </div>
                                ) : (
                                    <div className="borrowed-grid">
                                        {borrowingInfo.borrowed_books.map(book => (
                                            <div key={book.transaction_id} className={`borrowed-book-card ${book.is_overdue ? 'overdue' : ''}`}>
                                                <div className="book-info">
                                                    <h4>{book.title}</h4>
                                                    <p className="due-date">Hạn trả: {new Date(book.due_date).toLocaleDateString('vi-VN')}</p>
                                                </div>
                                                <div className="item-status">
                                                    {book.is_overdue ? 
                                                        <span className="status-badge danger">Quá hạn {Math.abs(book.days_left)} ngày</span> : 
                                                        <span className="status-badge success">Còn {book.days_left} ngày</span>
                                                    }
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'search' && (
                        <div className="search-tab">
                            <div className="search-bar">
                                <input 
                                    type="text" 
                                    placeholder="Tìm tên sách, tác giả..." 
                                    value={bookSearch}
                                    onChange={(e) => setBookSearch(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && fetchBooks()}
                                />
                                <button className="btn btn-primary" onClick={fetchBooks}>Tìm kiếm</button>
                            </div>

                            <div className="books-grid">
                                {isLoading ? (
                                    <div className="loading">Đang tìm kiếm...</div>
                                ) : books.length === 0 ? (
                                    <div className="empty-state">Không tìm thấy sách nào.</div>
                                ) : (
                                    books.map(book => (
                                        <div key={book.book_id} className="book-catalog-card">
                                            <div className="book-cover">📖</div>
                                            <div className="book-details">
                                                <h4>{book.title}</h4>
                                                <p className="author">{book.author}</p>
                                                <span className={`status-pill ${book.status.toLowerCase()}`}>
                                                    {book.status === 'AVAILABLE' ? 'Sẵn sàng' : 'Đã được mượn'}
                                                </span>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'history' && (
                        <div className="history-tab">
                            {/* ... existing history tab content ... */}
                            <div className="history-list">
                                {isLoading ? (
                                    <div className="loading">Đang tải...</div>
                                ) : history.length === 0 ? (
                                    <div className="empty-state">Bạn chưa có lịch sử mượn trả sách.</div>
                                ) : (
                                    <table className="history-table">
                                        <thead>
                                            <tr>
                                                <th>Sách</th>
                                                <th>Ngày mượn</th>
                                                <th>Ngày trả</th>
                                                <th>Trạng thái</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {history.map(t => (
                                                <tr key={t.transaction_id}>
                                                    <td><strong>{t.book_title}</strong></td>
                                                    <td>{new Date(t.borrow_date).toLocaleDateString('vi-VN')}</td>
                                                    <td>{t.return_date ? new Date(t.return_date).toLocaleDateString('vi-VN') : '-'}</td>
                                                    <td>
                                                        <span className={`status-pill ${t.status.toLowerCase()}`}>
                                                            {t.status === 'BORROWED' ? 'Đang mượn' : 'Đã trả'}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'chatbot' && (
                        <div className="chatbot-tab-content">
                            <AIChatbot fullPage={true} />
                        </div>
                    )}
                </section>
            </main>
        </div>
    )
}
