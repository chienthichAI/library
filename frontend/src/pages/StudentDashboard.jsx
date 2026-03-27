import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Sidebar from '../components/student/Sidebar';
import Overview from '../components/student/Overview';
import BookSearch from '../components/student/BookSearch';
import BookReturnScreen from './BookReturnScreen';
import AIChatbot from '../components/AIChatbot';
import './StudentDashboard.css';

const StudentDashboard = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState('overview'); // Default to overview
    const [student, setStudent] = useState(null);

    useEffect(() => {
        const storedId = sessionStorage.getItem('smartlib_student_id');
        const storedName = sessionStorage.getItem('smartlib_student_name');
        
        if (!storedId) {
            navigate('/verify');
            return;
        }

        setStudent({
            id: storedId,
            name: storedName,
            // Add other fields if needed
        });

        // If redirected with a specific tab
        if (location.state?.activeTab) {
            setActiveTab(location.state.activeTab);
        }
    }, [navigate, location]);

    const renderContent = () => {
        switch (activeTab) {
            case 'overview':
                return <Overview studentId={student.id} />;
            case 'search':
                return (
                    <div className="dashboard-content search-view animate-fade-in">
                        <div className="view-header">
                            <h1>🔍 Tìm kiếm sách</h1>
                            <p className="view-subtitle">Tìm kiếm hàng ngàn đầu sách trong thư viện SmartLib</p>
                        </div>
                        <div className="view-body">
                            <BookSearch />
                        </div>
                    </div>
                );
            case 'history':
                return (
                    <div className="dashboard-placeholder animate-fade-in">
                        <h2>📜 Lịch sử mượn trả</h2>
                        <p>Tính năng đang được phát triển...</p>
                    </div>
                );
            case 'ai':
                return (
                    <div className="dashboard-view ai-view animate-fade-in">
                        <div className="view-header">
                            <h1>Trợ lý AI SmartLib</h1>
                        </div>
                        <div className="view-content">
                            <AIChatbot isEmbedded={true} />
                        </div>
                    </div>
                );
            case 'borrow_return':
                return (
                    <div className="dashboard-view animate-fade-in">
                         {/* We reuse BookReturnScreen component logic here */}
                         <BookReturnScreen isEmbedded={true} />
                    </div>
                );
            default:
                return null;
        }
    };

    if (!student) return null;

    return (
        <div className="student-dashboard-layout">
            <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
            <main className="dashboard-main-content">
                {renderContent()}
            </main>
        </div>
    );
};

export default StudentDashboard;
