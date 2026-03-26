import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import WelcomeScreen from './pages/WelcomeScreen'
import RegistrationFlow from './pages/RegistrationFlow'
import VerificationScreen from './pages/VerificationScreen'
import BookReturnScreen from './pages/BookReturnScreen'
import AdminDashboard from './pages/AdminDashboard'
import BookDetail from './pages/BookDetail'
import AIChatbot from './components/AIChatbot'
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
          <Route path="/books/:id" element={<BookDetail />} />
        </Routes>
        <AIChatbot />
      </div>
    </BrowserRouter>
  )
}

export default App
