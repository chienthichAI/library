import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import './AIChatbot.css';

const AIChatbot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([
        { role: 'bot', content: 'Chào bạn! Mình là trợ lý SmartLib. Bạn cần tìm sách gì hay muốn mình tư vấn gì không?' }
    ]);
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (isOpen) scrollToBottom();
    }, [messages, loading, isOpen]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = input;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setLoading(true);

        try {
            const response = await axios.post(`${API_URL}/ai/chat`, {
                message: userMessage,
                context_type: 'search'
            });

            const { reply, suggestions, success } = response.data;

            if (success) {
                setMessages(prev => [...prev, {
                    role: 'bot',
                    content: reply,
                    suggestions: suggestions
                }]);
            } else {
                setMessages(prev => [...prev, {
                    role: 'bot',
                    content: 'Xin lỗi, mình đang gặp chút trục trặc. Bạn thử lại nhé!'
                }]);
            }
        } catch (error) {
            console.error('Chat error:', error);
            setMessages(prev => [...prev, {
                role: 'bot',
                content: 'Lỗi kết nối máy chủ AI. Vui lòng kiểm tra lại sau.'
            }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-bot-container">
            <button
                className="chat-toggle-btn"
                onClick={() => setIsOpen(!isOpen)}
            >
                {isOpen ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                ) : (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                )}
            </button>

            {isOpen && (
                <div className="chat-window">
                    <div className="chat-header">
                        <div className="bot-avatar">🤖</div>
                        <h3>SmartLib Assistant</h3>
                    </div>

                    <div className="chat-messages">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`message ${msg.role === 'bot' ? 'bot-message' : 'user-message'}`}>
                                {msg.content}

                                {msg.suggestions && msg.suggestions.length > 0 && (
                                    <div className="suggestions-container">
                                        {msg.suggestions.map((book, bIdx) => (
                                            <div key={bIdx} className="book-card" onClick={() => navigate(`/books/${book.book_id}`)}>
                                                <h4>{book.title}</h4>
                                                <p>{book.author}</p>
                                                <span className={`status-badge ${book.status === 'AVAILABLE' ? 'status-available' : 'status-borrowed'}`}>
                                                    {book.status === 'AVAILABLE' ? 'Sẵn sàng' : 'Đã mượn'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                        {loading && (
                            <div className="message bot-message">
                                <div className="loading-dots">
                                    <div className="dot"></div>
                                    <div className="dot"></div>
                                    <div className="dot"></div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="chat-input-area">
                        <input
                            type="text"
                            placeholder="Nhập yêu cầu tìm sách..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        />
                        <button className="send-btn" onClick={handleSend} disabled={loading}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIChatbot;
