import { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import FaceCapture from '../components/FaceCapture'
import StudentInfoForm from '../components/StudentInfoForm'
import RegistrationComplete from '../components/RegistrationComplete'
import { API_URL } from '../config'
import './RegistrationFlow.css'

const STEPS = [
    { id: 1, label: 'Chụp khuôn mặt' },
    { id: 2, label: 'Thông tin cá nhân' },
    { id: 3, label: 'Hoàn thành' }
]

export default function RegistrationFlow() {
    const navigate = useNavigate()
    const [currentStep, setCurrentStep] = useState(1)
    const [faceImages, setFaceImages] = useState([])
    const [studentInfo, setStudentInfo] = useState(null)
    const [registrationResult, setRegistrationResult] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleFaceCapture = (images) => {
        setFaceImages(images)
        setCurrentStep(2)
    }

    const handleStudentInfo = async (info) => {
        setStudentInfo(info)
        setIsLoading(true)
        setError(null)

        try {
            // Step 1: Create student account
            const studentResponse = await fetch(`${API_URL}/students/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    student_id: info.studentId,
                    full_name: info.fullName,
                    email: info.email,
                    phone: info.phone
                })
            })

            if (!studentResponse.ok) {
                const err = await studentResponse.json()
                throw new Error(err.detail || 'Không thể tạo tài khoản sinh viên')
            }

            const student = await studentResponse.json()

            // Step 2: Register face embeddings (using all captured images for robust recognition)
            let registeredCount = 0;
            let lastError = null;

            for (let i = 0; i < faceImages.length; i++) {
                const formData = new FormData()
                formData.append('student_id', info.studentId)
                formData.append('image', faceImages[i].blob)

                try {
                    const faceResponse = await fetch(`${API_URL}/auth/register-face`, {
                        method: 'POST',
                        body: formData
                    })

                    if (!faceResponse.ok) {
                        const err = await faceResponse.json()
                        lastError = err.detail || 'Không thể đăng ký một góc khuôn mặt'
                        console.warn(`Face registration failed for image ${i + 1}:`, lastError)
                    } else {
                        registeredCount++;
                    }
                } catch (e) {
                    lastError = e.message;
                    console.warn(`Face registration fetch failed for image ${i + 1}:`, e)
                }
            }

            if (registeredCount === 0) {
                throw new Error(`Đăng ký khuôn mặt thất bại: ${lastError}`)
            }

            setRegistrationResult({
                success: true,
                student: student,
                message: 'Đăng ký thành công!'
            })
            setCurrentStep(3)

        } catch (err) {
            console.error('Registration error:', err)
            setError(err.message)
        } finally {
            setIsLoading(false)
        }
    }

    const handleBack = () => {
        if (currentStep > 1) {
            setCurrentStep(currentStep - 1)
            setError(null)
        } else {
            navigate('/')
        }
    }

    const handleFinish = () => {
        navigate('/')
    }

    return (
        <div className="registration-flow">
            {/* Header */}
            <header className="reg-header">
                <button className="back-btn" onClick={handleBack}>
                    ← Quay lại
                </button>
                <h2>Đăng ký sinh viên mới</h2>
                <div className="spacer"></div>
            </header>

            {/* Progress Steps */}
            <div className="steps">
                {STEPS.map((step) => (
                    <div
                        key={step.id}
                        className={`step ${currentStep === step.id ? 'active' : ''} ${currentStep > step.id ? 'completed' : ''}`}
                    >
                        <div className="step-number">
                            {currentStep > step.id ? '✓' : step.id}
                        </div>
                        <span className="step-label">{step.label}</span>
                    </div>
                ))}
            </div>

            {/* Main Content */}
            <main className="reg-content">
                {error && (
                    <div className="error-banner animate-fade-in">
                        <span>⚠️</span> {error}
                        <button onClick={() => setError(null)}>×</button>
                    </div>
                )}

                {currentStep === 1 && (
                    <FaceCapture
                        onCapture={handleFaceCapture}
                        requiredCaptures={3}
                    />
                )}

                {currentStep === 2 && (
                    <StudentInfoForm
                        onSubmit={handleStudentInfo}
                        isLoading={isLoading}
                        capturedImages={faceImages}
                    />
                )}

                {currentStep === 3 && (
                    <RegistrationComplete
                        result={registrationResult}
                        onFinish={handleFinish}
                    />
                )}
            </main>
        </div>
    )
}
