import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { chatApi } from '../services/chatbotApi';
import './AIChatbot.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const AIChatbot = () => {
    const navigate = useNavigate();
    const [isOpen, setIsOpen] = useState(false);

    const [messages, setMessages] = useState([
        {
            role: 'ai',
            content: 'Xin chào! 👋 Tôi là trợ lý AI thư viện SmartLib. Tôi có thể giúp bạn tìm sách, hướng dẫn quy trình mượn/trả hoặc giải đáp thắc mắc về thư viện!'
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState(null);
    const [showUpload, setShowUpload] = useState(false);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        const currentInput = input;
        setInput('');
        setIsLoading(true);

        try {
            const studentId = sessionStorage.getItem('smartlib_student_id');
            const response = await chatApi.sendMessage(currentInput, undefined, studentId);
            const { answer, suggestions } = response;
            setMessages(prev => [...prev, { 
                role: 'ai', 
                content: answer,
                suggestions: suggestions
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'error',
                content: '⚠️ Không thể kết nối đến server. Vui lòng thử lại sau.'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        setUploadStatus(null);

        try {
            const response = await chatApi.uploadDocument(file);
            setUploadStatus({
                type: 'success',
                message: `✅ ${response.message} (${response.chunks_created} chunks)`
            });
            setMessages(prev => [...prev, {
                role: 'ai',
                content: `📄 Tài liệu "${file.name}" đã được xử lý thành công! Bạn có thể hỏi tôi về nội dung bên trong.`
            }]);
        } catch (error) {
            setUploadStatus({
                type: 'error',
                message: '❌ Upload thất bại. Vui lòng thử lại.'
            });
        } finally {
            setIsUploading(false);
            setShowUpload(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const quickQuestions = [
        '📋 Quy trình mượn sách',
        '🔄 Thủ tục trả sách',
        '🕒 Giờ mở cửa thư viện',
        '💳 Số lượng sách tối đa'
    ];

    const handleQuickQuestion = (q) => {
        const cleanQ = q.replace(/^[^\s]+ /, '');
        setInput(cleanQ);
    };

    return (
        <div className="ai-chatbot-wrapper">
            {/* Floating Toggle Button */}
            <button
                id="chatbot-toggle-btn"
                className={`chatbot-toggle ${isOpen ? 'active' : ''}`}
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Toggle AI Chatbot"
            >
                <span className="toggle-icon">
                    {isOpen ? (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                    ) : (
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                            <circle cx="9" cy="10" r="1" fill="currentColor" />
                            <circle cx="12" cy="10" r="1" fill="currentColor" />
                            <circle cx="15" cy="10" r="1" fill="currentColor" />
                        </svg>
                    )}
                </span>
                {!isOpen && <span className="toggle-pulse" />}
            </button>

            {/* Chat Window */}
            {isOpen && (
                <div className="chatbot-window" id="chatbot-window">
                    {/* Header */}
                    <div className="chatbot-header">
                        <div className="header-left">
                            <div className="ai-avatar">
                                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 2a4 4 0 0 1 4 4v2H8V6a4 4 0 0 1 4-4z" />
                                    <rect x="3" y="8" width="18" height="12" rx="3" />
                                    <circle cx="9" cy="14" r="1.5" fill="currentColor" />
                                    <circle cx="15" cy="14" r="1.5" fill="currentColor" />
                                </svg>
                            </div>
                            <div className="header-info">
                                <h3>SmartLib AI</h3>
                                <span className="status-dot" />
                                <span className="status-text">Online</span>
                            </div>
                        </div>
                        <div className="header-actions">
                            <button
                                className="header-btn clear-btn"
                                onClick={() => {
                                    chatApi.clearSession();
                                    setMessages([{
                                        role: 'ai',
                                        content: 'Phiên chat đã được làm mới. Tôi có thể giúp gì cho bạn?'
                                    }]);
                                }}
                                title="Xóa lịch sử chat"
                                id="chatbot-clear-btn"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                    <path d="M3 3v5h5" />
                                </svg>
                            </button>
                            <button
                                className="header-btn upload-trigger"
                                onClick={() => setShowUpload(!showUpload)}
                                title="Upload tài liệu"
                                id="upload-toggle-btn"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                                </svg>
                            </button>
                            <button
                                className="header-btn close-btn"
                                onClick={() => setIsOpen(false)}
                                title="Đóng"
                                id="chatbot-close-btn"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18" />
                                    <line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            </button>
                        </div>
                    </div>

                    {/* Upload Area */}
                    {showUpload && (
                        <div className="upload-area">
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,.csv,.txt,.docx"
                                onChange={handleFileUpload}
                                id="doc-upload-input"
                                hidden
                            />
                            <button
                                className="upload-btn"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isUploading}
                                id="doc-upload-btn"
                            >
                                {isUploading ? (
                                    <>
                                        <span className="upload-spinner" />
                                        Đang xử lý...
                                    </>
                                ) : (
                                    <>
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                            <polyline points="14 2 14 8 20 8" />
                                            <line x1="12" y1="18" x2="12" y2="12" />
                                            <line x1="9" y1="15" x2="12" y2="12" />
                                            <line x1="15" y1="15" x2="12" y2="12" />
                                        </svg>
                                        Upload tài liệu (PDF, CSV, TXT, DOCX)
                                    </>
                                )}
                            </button>
                            {uploadStatus && (
                                <div className={`upload-status ${uploadStatus.type}`}>
                                    {uploadStatus.message}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Messages */}
                    <div className="chatbot-messages" id="chatbot-messages">
                        {messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={`chat-bubble ${msg.role === 'ai' ? 'ai' : 
                                           msg.role === 'bot' ? 'ai' : 
                                           msg.role === 'user' ? 'user' : 'error'}`}
                            >
                                {(msg.role === 'ai' || msg.role === 'bot') && (
                                    <div className="bubble-avatar">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                            <circle cx="12" cy="12" r="10" />
                                            <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                                            <line x1="9" y1="9" x2="9.01" y2="9" />
                                            <line x1="15" y1="9" x2="15.01" y2="9" />
                                        </svg>
                                    </div>
                                )}
                                <div className="bubble-content">
                                    {(msg.role === 'ai' || msg.role === 'bot') ? (
                                        <div className="prose">
                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                {msg.content}
                                            </ReactMarkdown>
                                        </div>
                                    ) : (
                                        <div className="user-text">
                                            {msg.content}
                                        </div>
                                    )}

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
                            </div>
                        ))}

                        {isLoading && (
                            <div className="chat-bubble ai">
                                <div className="bubble-avatar">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                        <circle cx="12" cy="12" r="10" />
                                        <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                                        <line x1="9" y1="9" x2="9.01" y2="9" />
                                        <line x1="15" y1="9" x2="15.01" y2="9" />
                                    </svg>
                                </div>
                                <div className="bubble-content typing-indicator">
                                    <span className="typing-dot" />
                                    <span className="typing-dot" />
                                    <span className="typing-dot" />
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Quick Questions (only if few messages) */}
                    {messages.length <= 1 && !isLoading && (
                        <div className="quick-questions">
                            {quickQuestions.map((q, i) => (
                                <button
                                    key={i}
                                    className="quick-q-btn"
                                    onClick={() => handleQuickQuestion(q)}
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Input Area */}
                    <div className="chatbot-input-area">
                        <input
                            type="text"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Nhập câu hỏi..."
                            disabled={isLoading}
                            id="chatbot-input"
                            autoComplete="off"
                        />
                        <button
                            className="send-btn"
                            onClick={handleSend}
                            disabled={isLoading || !input.trim()}
                            id="chatbot-send-btn"
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13" />
                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIChatbot;
