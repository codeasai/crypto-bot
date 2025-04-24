import React from 'react';
import { Box, Typography, Container, Paper } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import Auth from '../components/Auth';

const Home = () => {
  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
    >
      <Container maxWidth="sm">
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Welcome to Crypto Bot
          </Typography>
          <Typography variant="body1" paragraph>
            Please sign in to continue
          </Typography>
          <Box sx={{ mt: 2 }}>
            <Auth />
          </Box>
        </Paper>
      </Container>
    </Box>
  );
};

export default Home; 