import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// สร้าง axios instance
const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Interceptor สำหรับเพิ่ม Authorization header
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Interceptor สำหรับจัดการ error
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    throw error;
  }
);

// --- Authentication --- (คงเดิม)
export const googleLogin = async (credentialResponse) => {
  try {
    const response = await api.post('/api/auth/google', {
      credential: credentialResponse.credential
    });
    return response.data; // คืนค่า response ทั้งหมดเพื่อให้ context จัดการ
  } catch (error) {
    console.error('Google login API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Google login failed');
  }
};

// --- Strategies --- (คงเดิม)
export const getStrategies = async () => {
  try {
    const response = await api.get('/api/strategies');
    if (!response.data.success) {
      throw new Error(response.data.error || response.data.message);
    }
    return response.data.strategies || [];
  } catch (error) {
    const errorMessage = error.response?.data?.message || error.message || 'ไม่สามารถดึงข้อมูลกลยุทธ์ได้';
    throw new Error(errorMessage);
  }
};

// --- Dashboard/Status Data --- (เพิ่มหรือแก้ไข)
export const getStatus = async () => {
  try {
    const response = await api.get('/api/status');
    return response.data;
  } catch (error) {
    console.error('Get status API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to fetch status');
  }
};

export const getPrice = async (symbol) => {
  try {
    const response = await api.get(`/api/price?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Get price API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to fetch price');
  }
};

export const getPortfolio = async () => {
  try {
    const response = await api.get('/api/portfolio');
    return response.data;
  } catch (error) {
    console.error('Get portfolio API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to fetch portfolio');
  }
};

export const getOrders = async (symbol) => {
  try {
    const response = await api.get(`/api/orders?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Get orders API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to fetch orders');
  }
};

// --- Bot Management --- (เพิ่มใหม่)
export const createBot = async (config) => {
  try {
    // ส่ง config ทั้งหมดไปเลย เพราะ backend รับ config เป็น object
    const response = await api.post('/api/bot/create', config);
    return response.data;
  } catch (error) {
    console.error('Create bot API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to create bot');
  }
};

export const deleteBot = async (botId) => {
  try {
    const response = await api.delete(`/api/bot/delete/${botId}`);
    return response.data;
  } catch (error) {
    console.error('Delete bot API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to delete bot');
  }
};

export const startBot = async (botId) => {
  try {
    const response = await api.post(`/api/bot/start/${botId}`);
    return response.data;
  } catch (error) {
    console.error('Start bot API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to start bot');
  }
};

export const stopBot = async (botId) => {
  try {
    const response = await api.post(`/api/bot/stop/${botId}`);
    return response.data;
  } catch (error) {
    console.error('Stop bot API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to stop bot');
  }
};

export const updateBotConfig = async (botId, config) => {
  try {
    const response = await api.post(`/api/bot/config/${botId}`, config);
    return response.data;
  } catch (error) {
    console.error('Update bot config API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to update bot config');
  }
};

// --- User Profile --- (เพิ่มใหม่)
export const updateUserProfile = async (updates) => {
  try {
    const response = await api.post('/api/user/update', updates);
    return response.data;
  } catch (error) {
    console.error('Update user profile API error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to update profile');
  }
}; 