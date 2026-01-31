import './RegistrationComplete.css'

export default function RegistrationComplete({ result, onFinish }) {
    return (
        <div className="registration-complete animate-fade-in">
            <div className="success-icon">
                <span>✓</span>
            </div>

            <h2>Đăng ký thành công!</h2>

            <p className="success-message">
                Chào mừng <strong>{result?.student?.full_name}</strong> đến với SmartLib
            </p>

            <div className="student-card">
                <div className="card-header">
                    <span className="card-icon">🎓</span>
                    <span>Thẻ sinh viên</span>
                </div>
                <div className="card-body">
                    <div className="card-row">
                        <span className="label">Mã sinh viên</span>
                        <span className="value">{result?.student?.student_id}</span>
                    </div>
                    <div className="card-row">
                        <span className="label">Họ và tên</span>
                        <span className="value">{result?.student?.full_name}</span>
                    </div>
                    {result?.student?.email && (
                        <div className="card-row">
                            <span className="label">Email</span>
                            <span className="value">{result?.student?.email}</span>
                        </div>
                    )}
                </div>
            </div>

            <div className="next-steps">
                <h4>Bước tiếp theo</h4>
                <ul>
                    <li>✓ Khuôn mặt của bạn đã được đăng ký vào hệ thống</li>
                    <li>📚 Bạn có thể mượn và trả sách bằng khuôn mặt</li>
                    <li>⏰ Thời hạn mượn sách: 14 ngày</li>
                </ul>
            </div>

            <button className="btn btn-success btn-large" onClick={onFinish}>
                Về trang chủ
            </button>
        </div>
    )
}
