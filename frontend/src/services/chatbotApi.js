// src/services/chatbotApi.js
// Service layer connect with RAG FastAPI Endpoints

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Helper to get or create a session ID for the current browser session.
 * Uses sessionStorage to maintain continuity across refreshes in the same tab.
 */
const getSessionId = () => {
    let sessionId = sessionStorage.getItem('smartlib_chat_session');
    if (!sessionId) {
        sessionId = `sess_${Math.random().toString(36).substr(2, 9)}_${Date.now()}`;
        sessionStorage.setItem('smartlib_chat_session', sessionId);
    }
    return sessionId;
};

export const chatApi = {
    // 1. Send query to Rag Pipeline for generation
    sendMessage: async (message, sessionId = getSessionId(), studentId = null, metadata = {}) => {
        try {
            const token = sessionStorage.getItem('smartlib_verification_token');
            const response = await fetch(`${BASE_URL}/api/chatbot/chat`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                body: JSON.stringify({ 
                    query: message,
                    session_id: sessionId,
                    student_id: studentId,
                    metadata: metadata
                })
            });
            if (!response.ok) throw new Error(`API Connection Failed: ${response.status}`);
            return response.json();
        } catch (error) {
            console.error("Chat API error:", error);
            throw error;
        }
    },
    
    // 2. Clear current session
    clearSession: () => {
        sessionStorage.removeItem('smartlib_chat_session');
    },

    // 3. Fetch session history
    getHistory: async (sessionId = getSessionId()) => {
        try {
            const token = sessionStorage.getItem('smartlib_verification_token');
            const response = await fetch(`${BASE_URL}/api/chatbot/history?session_id=${sessionId}`, {
                method: 'GET',
                headers: { 
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                }
            });
            if (!response.ok) throw new Error(`Failed to fetch history: ${response.status}`);
            return response.json();
        } catch (error) {
            console.error("Fetch history error:", error);
            throw error;
        }
    },
    
    // 4. Upload doc to trigger ingestion pipeline
    uploadDocument: async (file) => {
        const formData = new FormData();
        formData.append("file", file);
        const token = sessionStorage.getItem('smartlib_verification_token');
        try {
            const response = await fetch(`${BASE_URL}/api/chatbot/upload-docs`, {
                method: 'POST',
                headers: {
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                body: formData
            });
            return response.json();
        } catch (error) {
            console.error("Upload API error:", error);
            throw error;
        }
    }
};
