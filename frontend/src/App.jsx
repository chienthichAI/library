import { useState } from 'react'

import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import WelcomeScreen from './pages/WelcomeScreen'
import RegistrationFlow from './pages/RegistrationFlow'
import VerificationScreen from './pages/VerificationScreen'
import BookReturnScreen from './pages/BookReturnScreen'
import AdminDashboard from './pages/AdminDashboard'
import BookDetail from './pages/BookDetail'
import AIChatbot from './components/AIChatbot'
import './App.css'

import StudentDashboard from './pages/StudentDashboard'

function AppContent() {
  const location = useLocation();
  
  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<WelcomeScreen />} />
        <Route path="/register" element={<RegistrationFlow />} />
        <Route path="/verify" element={<VerificationScreen />} />
        <Route path="/return" element={<BookReturnScreen />} />
        <Route path="/admin" element={<AdminDashboard />} />
        <Route path="/dashboard_student" element={<StudentDashboard />} />
        <Route path="/books/:id" element={<BookDetail />} />
      </Routes>
      {/* Hide global floating chatbot on student dashboard to avoid duplication */}
      {!location.pathname.includes('dashboard_student') && 
       !location.pathname.includes('admin') && <AIChatbot />}
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}

export default App
