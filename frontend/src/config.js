// Frontend API Configuration
// This file centralizes the API URL configuration

// Use environment variable in production, fallback to localhost for development
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// For debugging - log API URL in development
if (import.meta.env.DEV) {
    console.log('API URL:', API_URL);
}
