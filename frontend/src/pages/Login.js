import React, { useState } from 'react';
import { Box, Container, Paper, Typography, Alert } from '@mui/material';
import { GoogleLogin } from '@react-oauth/google';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState(null);

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      setError(null);
      const success = await login(credentialResponse);
      if (success) {
        navigate('/dashboard');
      }
    } catch (error) {
      setError(error.message || 'การเข้าสู่ระบบล้มเหลว');
      console.error('Login failed:', error);
    }
  };

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Paper elevation={3} sx={{ p: 4, width: '100%' }}>
          <Typography component="h1" variant="h5" align="center" gutterBottom>
            เข้าสู่ระบบ
          </Typography>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => setError('การเข้าสู่ระบบด้วย Google ล้มเหลว')}
            />
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Login; 