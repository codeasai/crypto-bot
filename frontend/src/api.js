import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const getPortfolioData = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/portfolio`);
    return response.data;
  } catch (error) {
    console.error('Error fetching portfolio data:', error);
    throw error;
  }
};

export const getActiveBots = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/bots/active`);
    return response.data;
  } catch (error) {
    console.error('Error fetching active bots:', error);
    throw error;
  }
};

export const createBot = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/api/bots`, data);
    return response.data;
  } catch (error) {
    console.error('Error creating bot:', error);
    throw error;
  }
}; 