import React, { useState } from 'react';
import { 
    LayoutDashboard, 
    Search, 
    History, 
    Bot, 
    ScanLine, 
    ChevronLeft, 
    ChevronRight,
    LogOut,
    UserCircle
} from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ activeTab, onTabChange }) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    
    // Get student info from session
    const studentName = sessionStorage.getItem('smartlib_student_name') || 'Sinh viên';
    const studentId = sessionStorage.getItem('smartlib_student_id') || 'STUDENT_ID';

    const menuItems = [
        { id: 'overview', label: 'Tổng quan', icon: LayoutDashboard },
        { id: 'search', label: 'Tìm sách', icon: Search },
        { id: 'history', label: 'Lịch sử mượn', icon: History },
        { id: 'ai', label: 'Trợ lý AI', icon: Bot, isSpecial: true },
        { id: 'borrow_return', label: 'Mượn & Trả sách', icon: ScanLine },
    ];

    return (
        <div className={`student-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
            {/* Toggle Button */}
            <button 
                className="toggle-sidebar-btn" 
                onClick={() => setIsCollapsed(!isCollapsed)}
            >
                {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </button>

            {/* User Profile */}
            <div className="sidebar-profile">
                <div className="profile-avatar">
                   <UserCircle size={isCollapsed ? 28 : 40} strokeWidth={1.5} />
                </div>
                {!isCollapsed && (
                    <div className="profile-info animate-fade-in">
                        <h3 className="student-name">{studentName}</h3>
                        <span className="student-id">ID: {studentId}</span>
                        <div className="student-rank">SINH VIÊN</div>
                    </div>
                )}
            </div>

            {/* Menu Items */}
            <nav className="sidebar-nav">
                {menuItems.map((item) => {
                    const Icon = item.icon;
                    return (
                        <button
                            key={item.id}
                            className={`nav-item ${activeTab === item.id ? 'active' : ''} ${item.isSpecial ? 'special-item' : ''}`}
                            onClick={() => onTabChange(item.id)}
                            title={isCollapsed ? item.label : ''}
                        >
                            <span className="nav-icon">
                                <Icon size={22} />
                            </span>
                            {!isCollapsed && <span className="nav-label">{item.label}</span>}
                            {activeTab === item.id && !isCollapsed && <div className="active-indicator" />}
                        </button>
                    );
                })}
            </nav>

            {/* Bottom Section */}
            <div className="sidebar-bottom">
                <button 
                    className="nav-item logout-btn" 
                    onClick={() => {
                        sessionStorage.clear();
                        window.location.href = '/';
                    }}
                >
                    <span className="nav-icon">
                        <LogOut size={22} />
                    </span>
                    {!isCollapsed && <span className="nav-label">Đăng xuất</span>}
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
