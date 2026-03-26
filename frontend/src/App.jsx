import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import WelcomeScreen from './pages/WelcomeScreen'
import RegistrationFlow from './pages/RegistrationFlow'
import VerificationScreen from './pages/VerificationScreen'
import BookReturnScreen from './pages/BookReturnScreen'
import AdminDashboard from './pages/AdminDashboard'
import StudentDashboard from './pages/StudentDashboard'
import BookDetail from './pages/BookDetail'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Routes>
          <Route path="/" element={<WelcomeScreen />} />
          <Route path="/register" element={<RegistrationFlow />} />
          <Route path="/verify" element={<VerificationScreen />} />
          <Route path="/return" element={<BookReturnScreen />} />
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/student" element={<StudentDashboard />} />
          <Route path="/books/:id" element={<BookDetail />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

export default App
