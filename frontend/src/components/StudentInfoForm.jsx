import { useState } from 'react'
import './StudentInfoForm.css'

export default function StudentInfoForm({ onSubmit, isLoading, capturedImages }) {
    const [formData, setFormData] = useState({
        studentId: '',
        fullName: '',
        email: '',
        phone: ''
    })
    const [errors, setErrors] = useState({})

    const handleChange = (e) => {
        const { name, value } = e.target
        setFormData(prev => ({ ...prev, [name]: value }))
        // Clear error when user types
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: null }))
        }
    }

    const validate = () => {
        const newErrors = {}

        if (!formData.studentId.trim()) {
            newErrors.studentId = 'Vui lòng nhập mã sinh viên'
        } else if (formData.studentId.trim().length < 4) {
            newErrors.studentId = 'Mã sinh viên quá ngắn'
        }

        if (!formData.fullName.trim()) {
            newErrors.fullName = 'Vui lòng nhập họ và tên'
        } else if (formData.fullName.trim().length < 3) {
            newErrors.fullName = 'Họ tên phải có ít nhất 3 ký tự'
        }

        if (formData.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
            newErrors.email = 'Email không hợp lệ'
        }

        if (formData.phone && !/^0\d{9,10}$/.test(formData.phone)) {
            newErrors.phone = 'Số điện thoại không hợp lệ'
        }

        setErrors(newErrors)
        return Object.keys(newErrors).length === 0
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        if (validate()) {
            onSubmit(formData)
        }
    }

    return (
        <div className="student-info-form animate-fade-in">
            <div className="form-header">
                <h3>Thông tin cá nhân</h3>
                <p>Điền thông tin để hoàn tất đăng ký</p>
            </div>

            {/* Captured face preview */}
            <div className="captured-preview">
                <div className="preview-label">Khuôn mặt đã chụp:</div>
                <div className="preview-images">
                    {capturedImages.map((img, idx) => (
                        <img key={idx} src={img.url} alt={`Captured ${idx + 1}`} />
                    ))}
                </div>
            </div>

            <form onSubmit={handleSubmit} className="info-form">
                <div className="form-group">
                    <label className="form-label" htmlFor="studentId">
                        Mã sinh viên <span className="required">*</span>
                    </label>
                    <input
                        type="text"
                        id="studentId"
                        name="studentId"
                        className={`form-input ${errors.studentId ? 'error' : ''}`}
                        placeholder="VD: FPT20240001"
                        value={formData.studentId}
                        onChange={handleChange}
                        autoFocus
                    />
                    {errors.studentId && <span className="error-text">{errors.studentId}</span>}
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="fullName">
                        Họ và tên <span className="required">*</span>
                    </label>
                    <input
                        type="text"
                        id="fullName"
                        name="fullName"
                        className={`form-input ${errors.fullName ? 'error' : ''}`}
                        placeholder="VD: Nguyễn Văn A"
                        value={formData.fullName}
                        onChange={handleChange}
                    />
                    {errors.fullName && <span className="error-text">{errors.fullName}</span>}
                </div>

                <div className="form-row">
                    <div className="form-group">
                        <label className="form-label" htmlFor="email">
                            Email
                        </label>
                        <input
                            type="email"
                            id="email"
                            name="email"
                            className={`form-input ${errors.email ? 'error' : ''}`}
                            placeholder="example@fpt.edu.vn"
                            value={formData.email}
                            onChange={handleChange}
                        />
                        {errors.email && <span className="error-text">{errors.email}</span>}
                    </div>

                    <div className="form-group">
                        <label className="form-label" htmlFor="phone">
                            Số điện thoại
                        </label>
                        <input
                            type="tel"
                            id="phone"
                            name="phone"
                            className={`form-input ${errors.phone ? 'error' : ''}`}
                            placeholder="0912345678"
                            value={formData.phone}
                            onChange={handleChange}
                        />
                        {errors.phone && <span className="error-text">{errors.phone}</span>}
                    </div>
                </div>

                <button
                    type="submit"
                    className="btn btn-primary btn-large submit-btn"
                    disabled={isLoading}
                >
                    {isLoading ? (
                        <>
                            <span className="spinner"></span>
                            Đang xử lý...
                        </>
                    ) : (
                        'Đăng ký'
                    )}
                </button>
            </form>
        </div>
    )
}
