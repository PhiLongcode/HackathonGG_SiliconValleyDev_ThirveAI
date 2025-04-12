import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
    'Content-Type': 'application/json',
  },
});
// const api = axios.create({
//    baseURL:'http://127.0.0.1:8000/process_audio'
//     'Content-Type':''

// })
// Add request interceptor to include auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;