import { useNavigate } from 'react-router-dom'
import './WelcomeScreen.css'

export default function WelcomeScreen() {
    const navigate = useNavigate()

    return (
        <div className="welcome-screen">
            <div className="welcome-content animate-fade-in">
                {/* Logo */}
                <div className="logo-container">
                    <div className="logo">
                        <span className="logo-icon">📚</span>
                    </div>
                    <h1>SmartLib</h1>
                    <p className="tagline">Hệ thống thư viện thông minh</p>
                </div>

                {/* Main Actions */}
                <div className="actions">
                    <button
                        className="action-card"
                        onClick={() => navigate('/verify')}
                    >
                        <div className="action-icon">👤</div>
                        <div className="action-text">
                            <h3>Mượn / Trả sách</h3>
                            <p>Đã có tài khoản</p>
                        </div>
                        <span className="action-arrow">→</span>
                    </button>

                    <button
                        className="action-card register"
                        onClick={() => navigate('/register')}
                    >
                        <div className="action-icon">✨</div>
                        <div className="action-text">
                            <h3>Đăng ký mới</h3>
                            <p>Tạo tài khoản sinh viên</p>
                        </div>
                        <span className="action-arrow">→</span>
                    </button>
                </div>

                {/* Footer */}
                <div className="welcome-footer">
                    <p>FPT University Library</p>
                </div>
            </div>

            {/* Background decoration */}
            <div className="bg-decoration">
                <div className="circle circle-1"></div>
                <div className="circle circle-2"></div>
                <div className="circle circle-3"></div>
            </div>
        </div>
    )
}
