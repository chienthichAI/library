import React, { useState, useEffect } from 'react';
import { BookOpen, AlertCircle, CreditCard, Clock } from 'lucide-react';
import { API_URL } from '../../config';
import './Overview.css';

const Overview = ({ studentId }) => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch(`${API_URL}/students/${studentId}/borrowing-info`);
                const data = await response.json();
                setStats(data);
            } catch (error) {
                console.error('Error fetching student stats:', error);
            } finally {
                setLoading(false);
            }
        };

        if (studentId) fetchStats();
    }, [studentId]);

    if (loading) return (
        <div className="overview-loading">
            <div className="spinner-small"></div>
            <span>Đang tải dữ liệu...</span>
        </div>
    );

    if (!stats) return <div className="overview-error">Không thể tải dữ liệu.</div>;

    return (
        <div className="overview-container animate-fade-in">
            {/* Stats Header */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon-wrapper blue">
                        <BookOpen size={24} />
                    </div>
                    <div className="stat-info">
                        <span className="stat-label">Sách đang mượn</span>
                        <h2 className="stat-value">{stats.currently_borrowed} / {stats.max_books}</h2>
                    </div>
                    <div className="stat-progress">
                        <div 
                            className="progress-bar" 
                            style={{ width: `${(stats.currently_borrowed / stats.max_books) * 100}%` }}
                        />
                    </div>
                </div>

                <div className={`stat-card ${stats.fine_balance > 0 ? 'warning' : ''}`}>
                    <div className={`stat-icon-wrapper ${stats.fine_balance > 0 ? 'red' : 'green'}`}>
                        {stats.fine_balance > 0 ? <AlertCircle size={24} /> : <CreditCard size={24} />}
                    </div>
                    <div className="stat-info">
                        <span className="stat-label">Tiền nợ quá hạn</span>
                        <h2 className="stat-value">{stats.fine_balance.toLocaleString('vi-VN')} VND</h2>
                    </div>
                    {stats.fine_balance > 0 && <span className="stat-badge">Cần thanh toán</span>}
                </div>
            </div>

            {/* Borrowed Books Table */}
            <div className="borrowed-section modern-card">
                <div className="section-header">
                    <h3>📚 Danh sách sách đã mượn</h3>
                </div>
                
                {stats.borrowed_books.length === 0 ? (
                    <div className="empty-state">
                        <p>Bạn chưa mượn cuốn sách nào.</p>
                    </div>
                ) : (
                    <div className="books-list">
                        {stats.borrowed_books.map((book, index) => (
                            <div key={index} className={`book-item ${book.is_overdue ? 'overdue' : ''}`}>
                                <div className="book-main">
                                    <div className="book-cover-placeholder">📖</div>
                                    <div className="book-text">
                                        <h4 className="title">{book.title}</h4>
                                        <span className="borrow-date">Mượn ngày: {new Date(book.borrow_date).toLocaleDateString('vi-VN')}</span>
                                    </div>
                                </div>
                                <div className="book-status-info">
                                    <div className="due-info">
                                        <Clock size={16} />
                                        <span>{book.is_overdue ? 'Quá hạn' : 'Còn lại'}: <strong>{Math.abs(book.days_left)} ngày</strong></span>
                                    </div>
                                    {book.fine_amount > 0 && (
                                        <span className="fine-tag">Phạt: {book.fine_amount.toLocaleString('vi-VN')}đ</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Overview;
