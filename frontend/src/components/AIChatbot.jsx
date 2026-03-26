import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { chatApi } from '../services/chatbotApi';
import './AIChatbot.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const AIChatbot = ({ fullPage = false, initialOpen = false }) => {
    const navigate = useNavigate();
    const [isOpen, setIsOpen] = useState(fullPage || initialOpen);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [userName, setUserName] = useState('');
    
    // Session Expiry Logic
    const [showExpiryModal, setShowExpiryModal] = useState(false);
    const [countdown, setCountdown] = useState(30);
    const timerRef = useRef(null);

    useEffect(() => {
        const token = sessionStorage.getItem('smartlib_verification_token');
        const name = sessionStorage.getItem('smartlib_student_name');
        
        setIsAuthenticated(!!token);
        if (name) setUserName(name);
        
        // If just opened and authenticated, fetch history
        if ((isOpen || fullPage) && !!token) {
            loadChatHistory();
        }
    }, [isOpen, fullPage]);

    // Countdown Logic for Session Expiry
    useEffect(() => {
        if (showExpiryModal && countdown > 0) {
            timerRef.current = setInterval(() => {
                setCountdown(prev => prev - 1);
            }, 1000);
        } else if (countdown === 0) {
            clearInterval(timerRef.current);
            navigate('/'); // Auto redirect to Home
        }
        return () => clearInterval(timerRef.current);
    }, [showExpiryModal, countdown, navigate]);

    const loadChatHistory = async () => {
        try {
            setIsLoading(true);
            const history = await chatApi.getHistory();
            if (history && history.length > 0) {
                // Map backend history (role: 'human'/'ai') to frontend (role: 'user'/'ai')
                const formattedHistory = history.map(h => ({
                    role: h.role === 'ai' ? 'ai' : 'user',
                    content: h.content,
                    suggestions: h.metadata?.suggestions || []
                }));
                setMessages(formattedHistory);
            } else {
                // No history, show personalized greeting
                const name = sessionStorage.getItem('smartlib_student_name') || '';
                setMessages([{
                    role: 'ai',
                    content: `Xin chào **${name}**! 👋 Tôi là SmartLib AI. Rất vui được gặp lại bạn. Tôi có thể giúp gì cho bạn hôm nay?`
                }]);
            }
        } catch (err) {
            console.error("Failed to load history:", err);
            // If unauthorized (401), trigger session expiry flow
            if (err.message?.includes('401')) {
                setShowExpiryModal(true);
                setCountdown(30);
            }
        } finally {
            setIsLoading(false);
        }
    };

    const initialAIMessage = {
        role: 'ai',
        content: userName 
            ? `Xin chào **${userName}**! 👋 Tôi là trợ lý AI SmartLib. Tôi có thể giúp gì cho bạn hôm nay?`
            : 'Xin chào! 👋 Tôi là trợ lý AI thư viện SmartLib. Vui lòng đăng nhập để tôi có thể hỗ trợ bạn tốt nhất!'
    };

    const [messages, setMessages] = useState([initialAIMessage]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState(null);
    const [showUpload, setShowUpload] = useState(false);
    const [showScanner, setShowScanner] = useState(false);
    const [scannerAction, setScannerAction] = useState({ target: '', title: '' });
    const [isScannerVideoReady, setIsScannerVideoReady] = useState(false);
    const [scanError, setScanError] = useState('');
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const canvasRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    // Start/stop camera stream for the anti-fraud barcode scan modal.
    useEffect(() => {
        if (!showScanner) {
            setIsScannerVideoReady(false);
            setScanError('');
            return;
        }

        let cancelled = false;

        const start = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                });

                if (cancelled) {
                    stream.getTracks().forEach((t) => t.stop());
                    return;
                }

                streamRef.current = stream;
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    await videoRef.current.play();
                }
            } catch (err) {
                console.error('Scanner camera error:', err);
                setScanError('Không thể mở camera. Vui lòng kiểm tra quyền truy cập camera trong trình duyệt.');
            }
        };

        start();

        return () => {
            cancelled = true;
            setIsScannerVideoReady(false);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((t) => t.stop());
                streamRef.current = null;
            }
        };
    }, [showScanner]);

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
            const { answer, suggestions, is_scanner_trigger, metadata } = response;
            
            // Check for scanner trigger (Anti-Fraud Flow)
            if (is_scanner_trigger && metadata?.requires_action === "SCAN_BOOK") {
                setScannerAction({
                    target: metadata.target_action,
                    title: metadata.book_title || "sách"
                });
                setShowScanner(true);
            }

            setMessages(prev => [...prev, { 
                role: 'ai', 
                content: answer,
                suggestions: suggestions
            }]);
        } catch (error) {
            console.error("Chat error:", error);
            if (error.message?.includes('401') || (error.response && error.response.status === 401)) {
                setShowExpiryModal(true);
                setCountdown(30);
                // Keep the last error message in chat for context
                setMessages(prev => [...prev, {
                    role: 'error',
                    content: '⚠️ Phiên xác thực khuôn mặt đã hết hạn. Vui lòng **[Xác thực lại](/verify)**.'
                }]);
            } else {
                setMessages(prev => [...prev, {
                    role: 'error',
                    content: '⚠️ Không thể kết nối đến server. Vui lòng thử lại sau.'
                }]);
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleScanComplete = async () => {
        setIsLoading(true);
        setScanError('');

        try {
            if (!videoRef.current || videoRef.current.readyState < 2) {
                setScanError('Camera chưa sẵn sàng. Vui lòng thử lại.');
                return;
            }

            // Snapshot current frame -> send to backend book detection
            const canvas = canvasRef.current;
            if (!canvas) {
                setScanError('Không thể chụp ảnh từ camera.');
                return;
            }

            const w = videoRef.current.videoWidth || 1280;
            const h = videoRef.current.videoHeight || 720;
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                setScanError('Không thể tạo canvas vẽ để chụp ảnh.');
                return;
            }
            ctx.drawImage(videoRef.current, 0, 0, w, h);

            const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
            if (!blob) {
                setScanError('Không thể mã hóa ảnh để nhận diện.');
                return;
            }

            const token = sessionStorage.getItem('smartlib_verification_token');
            const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
            const formData = new FormData();
            formData.append('image', blob, 'scan.jpg');

            const detectRes = await axios.post(
                `${BASE_URL}/api/v1/books/detect`,
                formData,
                {
                    headers: {
                        ...(token ? { Authorization: `Bearer ${token}` } : {})
                    }
                }
            );

            const detected = detectRes?.data;
            const detectedBarcode = detected?.barcode;

            if (!detected?.success || !detectedBarcode) {
                setScanError('Không nhận diện được mã vạch. Vui lòng đưa cuốn sách vào khung hình rõ hơn.');
                return;
            }

            // Close scanner UI only after we successfully detected a barcode.
            setShowScanner(false);
        
            const studentId = sessionStorage.getItem('smartlib_student_id');
            const actionText = scannerAction.target === 'borrow_book' ? 'mượn' : 'trả';
            const query = `Tôi đã quét mã sách ${scannerAction.title}. Hãy thực hiện ${actionText}.`;
            
            const response = await chatApi.sendMessage(query, undefined, studentId, { 
                verified_barcode: detectedBarcode,
                target_action: scannerAction.target
            });
            
            setMessages(prev => [...prev, { 
                role: 'ai', 
                content: response.answer,
                suggestions: response.suggestions
            }]);
        } catch (error) {
            console.error("Scan confirm error:", error);
            setScanError('❌ Lỗi hệ thống khi nhận diện mã vạch. Vui lòng thử lại.');
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
        <div className={`ai-chatbot-wrapper ${fullPage ? 'full-page' : ''}`}>
            {/* Floating Toggle Button (Hidden in full-page) */}
            {!fullPage && (
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
            )}

            {/* Chat Window */}
            {(isOpen || fullPage) && (
                <div className={`chatbot-window ${fullPage ? 'full-page' : ''}`} id="chatbot-window">
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
                                <h3>SmartLib AI Assistant</h3>
                                <div className="header-status">
                                    <span className="status-dot" />
                                    <span className="status-text">
                                        {isAuthenticated && userName ? `Phục vụ: ${userName}` : 'Online'}
                                    </span>
                                </div>
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
                            {!fullPage && (
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
                            )}
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

                    {/* Unified Auth + Message View */}
                    {!isAuthenticated ? (
                        <div className="locked-chatbot-container">
                            <div className="locked-content">
                                <div className="lock-icon">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                                    </svg>
                                </div>
                                <h3>Yêu Cầu Xác Thực</h3>
                                <p>Để bảo mật thông tin và cá nhân hóa trải nghiệm, vui lòng xác thực danh tính của bạn.</p>
                                <button className="auth-redirect-btn" onClick={() => { if(!fullPage) setIsOpen(false); navigate('/verify'); }}>
                                    Xác Thực Ngay
                                </button>
                            </div>
                        </div>
                    ) : (
                        <>
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
                                    placeholder="Hỏi AI thư viện..."
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
                        </>
                    )}
                </div>
            )}

            {/* Book Scanner Overlay */}
            {showScanner && (
                <div className="scanner-overlay">
                    <div className="scanner-card">
                        <div className="scanner-header">
                            <h4>Quét Sách Hệ Thống</h4>
                            <button className="close-scanner" onClick={() => setShowScanner(false)}>&times;</button>
                        </div>
                        <div className="scanner-viewport">
                            <video
                                ref={videoRef}
                                className="scanner-video"
                                autoPlay
                                playsInline
                                muted
                                onCanPlay={() => setIsScannerVideoReady(true)}
                            />
                            <div className="scanner-laser" />
                            <div className="scanner-tips">
                                <p>Hướng mã vạch của cuốn <strong>{scannerAction.title}</strong> vào khung hình</p>
                            </div>
                            {!isScannerVideoReady && (
                                <div className="scanner-cam-placeholder">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                                        <circle cx="12" cy="13" r="4" />
                                    </svg>
                                </div>
                            )}
                        </div>
                        {scanError && <div className="scan-error-text">{scanError}</div>}
                        <button className="scan-confirm-btn" onClick={() => handleScanComplete()}>
                            Xác nhận đã quét
                        </button>
                        <canvas ref={canvasRef} style={{ display: 'none' }} />
                    </div>
                </div>
            )}

            {/* Session Expiry Warning Modal */}
            {showExpiryModal && (
                <div className="expiry-modal-overlay">
                    <div className="expiry-modal-card">
                        <div className="expiry-icon">🔒</div>
                        <h3>Phiên Làm Việc Hết Hạn</h3>
                        <p>Phiên xác thực khuôn mặt của bạn đã hết hạn bảo mật.</p>
                        <div className="expiry-countdown-circle">
                            <span className="countdown-number">{countdown}s</span>
                        </div>
                        <p className="expiry-tip">Tự động quay lại Trang chủ sau {countdown}s</p>
                        <div className="expiry-actions">
                            <button 
                                className="expiry-btn-reauth"
                                onClick={() => navigate('/verify')}
                            >
                                Xác thực ngay
                            </button>
                            <button 
                                className="expiry-btn-home"
                                onClick={() => navigate('/')}
                            >
                                Quay về Home
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIChatbot;
