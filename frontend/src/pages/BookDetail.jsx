import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { API_URL } from '../config'
import './BookDetail.css'

export default function BookDetail() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [book, setBook] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        const fetchBook = async () => {
            try {
                setLoading(true)
                const response = await fetch(`${API_URL}/books/${id}`)
                if (!response.ok) {
                    throw new Error('Không tìm thấy thông tin sách')
                }
                const data = await response.json()
                setBook(data)
            } catch (err) {
                console.error('Error fetching book:', err)
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }

        fetchBook()
    }, [id])

    if (loading) return <div className="book-detail-loading">Đang tải thông tin...</div>

    if (error || !book) {
        return (
            <div className="book-detail-error">
                <div className="error-card">
                    <h2>⚠️ Lỗi</h2>
                    <p>{error || 'Sách không tồn tại'}</p>
                    <button onClick={() => navigate(-1)}>Quay lại</button>
                </div>
            </div>
        )
    }

    return (
        <div className="book-detail-container animate-fade-in">
            <header className="detail-header">
                <button className="back-btn" onClick={() => navigate(-1)}>← Quay lại</button>
                <h1>Thông tin sách</h1>
            </header>

            <main className="detail-content">
                <div className="book-hero">
                    <div className="book-cover-placeholder">📖</div>
                    <div className="book-main-info">
                        <h2>{book.title}</h2>
                        <p className="author">Tác giả: {book.author}</p>
                        <div className={`status-badge ${book.status.toLowerCase()}`}>
                            {book.status === 'AVAILABLE' ? '✅ Sẵn sàng' : '📅 Đã mượn'}
                        </div>
                    </div>
                </div>

                <div className="book-specs">
                    <div className="spec-item">
                        <span className="label">Mã ISBN:</span>
                        <span className="value">{book.isbn_13 || book.book_id}</span>
                    </div>
                    <div className="spec-item">
                        <span className="label">Mã Barcode:</span>
                        <span className="value">{book.barcode || 'N/A'}</span>
                    </div>
                    <div className="spec-item">
                        <span className="label">Nhà xuất bản:</span>
                        <span className="value">{book.publisher || 'Thông tin đang cập nhật'}</span>
                    </div>
                    <div className="spec-item">
                        <span className="label">Năm xuất bản:</span>
                        <span className="value">{book.publication_year || 'N/A'}</span>
                    </div>
                    <div className="spec-item">
                        <span className="label">Ngôn ngữ:</span>
                        <span className="value">{book.language || 'Tiếng Việt'}</span>
                    </div>
                </div>

                {book.description && (
                    <div className="book-description">
                        <h3>Mô tả</h3>
                        <p>{book.description}</p>
                    </div>
                )}
            </main>
        </div>
    )
}
