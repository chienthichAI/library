import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { API_URL } from '../../config';
import './BookSearch.css';

const BookSearch = () => {
    const [query, setQuery] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [activeIndex, setVisibleIndex] = useState(-1);
    const navigate = useNavigate();
    const searchRef = useRef(null);
    const debounceTimer = useRef(null);

    // Close suggestions when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (searchRef.current && !searchRef.current.contains(event.target)) {
                setShowSuggestions(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const fetchSuggestions = async (searchTerm) => {
        if (searchTerm.length < 2) {
            setSuggestions([]);
            return;
        }

        setIsLoading(true);
        try {
            const response = await fetch(`${API_URL}/books?search=${encodeURIComponent(searchTerm)}&limit=6`);
            if (response.ok) {
                const data = await response.json();
                setSuggestions(data.books || []);
            }
        } catch (error) {
            console.error('Lỗi khi tìm kiếm sách:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleInputChange = (e) => {
        const value = e.target.value;
        setQuery(value);
        setShowSuggestions(true);
        setVisibleIndex(-1);

        if (debounceTimer.current) clearTimeout(debounceTimer.current);
        debounceTimer.current = setTimeout(() => {
            fetchSuggestions(value);
        }, 300);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setVisibleIndex(prev => (prev < suggestions.length - 1 ? prev + 1 : prev));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setVisibleIndex(prev => (prev > 0 ? prev - 1 : prev));
        } else if (e.key === 'Enter') {
            if (activeIndex >= 0 && suggestions[activeIndex]) {
                handleSelectBook(suggestions[activeIndex].book_id);
            }
        } else if (e.key === 'Escape') {
            setShowSuggestions(false);
        }
    };

    const handleSelectBook = (bookId) => {
        navigate(`/books/${bookId}`);
    };

    const clearSearch = () => {
        setQuery('');
        setSuggestions([]);
        setShowSuggestions(false);
    };

    return (
        <div className="book-search-container" ref={searchRef}>
            <div className={`search-input-wrapper ${showSuggestions && suggestions.length > 0 ? 'active' : ''}`}>
                <div className="search-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                </div>
                <input
                    type="text"
                    className="search-input"
                    placeholder="Tìm kiếm theo tên sách, tác giả hoặc mã sách..."
                    value={query}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    onFocus={() => query.length >= 2 && setShowSuggestions(true)}
                    autoComplete="off"
                />
                {query && (
                    <button className="clear-btn" onClick={clearSearch}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                    </button>
                )}
            </div>

            {showSuggestions && (query.length >= 2) && (
                <div className="suggestions-dropdown animate-slide-down">
                    {isLoading ? (
                        <div className="suggestion-status">
                            <span className="loader-dot" />
                            Đang tìm kiếm...
                        </div>
                    ) : suggestions.length > 0 ? (
                        <div className="suggestions-list">
                            {suggestions.map((book, index) => (
                                <div
                                    key={book.book_id}
                                    className={`suggestion-item ${index === activeIndex ? 'active' : ''}`}
                                    onClick={() => handleSelectBook(book.book_id)}
                                >
                                    <div className="suggestion-book-info">
                                        <div className="book-title-row">
                                            <span className="book-title">{book.title}</span>
                                            <span className={`status-badge ${book.status.toLowerCase()}`}>
                                                {book.status === 'AVAILABLE' ? 'Sẵn sàng' : 'Đã mượn'}
                                            </span>
                                        </div>
                                        <div className="book-author-row">
                                            <span className="book-author">{book.author}</span>
                                            <span className="book-id">ID: {book.book_id}</span>
                                        </div>
                                    </div>
                                    <div className="suggestion-arrow">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <polyline points="9 18 15 12 9 6" />
                                        </svg>
                                    </div>
                                </div>
                            ))}
                            <div className="suggestion-footer">
                                Nhấn Enter để xem chi tiết
                            </div>
                        </div>
                    ) : (
                        <div className="suggestion-status no-results">
                            Không tìm thấy sách phù hợp với "{query}"
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default BookSearch;
