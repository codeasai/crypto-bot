import React, { useState, useMemo, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Dashboard from './pages/Dashboard';
import Home from './pages/Home';
import Strategies from './pages/Strategies';
import AccountPage from './pages/AccountPage';
import BotControl from './pages/BotControl';
import MyBots from './pages/MyBots';
import Portfolio from './pages/Portfolio';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import './App.css';
import { AppBar, Toolbar, Typography, Button, Container, Box, CircularProgress } from '@mui/material';
import { GoogleOAuthProvider } from '@react-oauth/google';
import Login from './pages/Login';
import axios from 'axios';

const themeConfig = {
  palette: {
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
};

const AppContent = () => {
  const { user, logout } = useAuth();
  const [isDarkMode, setIsDarkMode] = React.useState(true);

  const theme = useMemo(
    () =>
      createTheme({
        ...themeConfig,
        palette: {
          ...themeConfig.palette,
          mode: isDarkMode ? 'dark' : 'light',
        },
      }),
    [isDarkMode],
  );

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
                Crypto Trading Bot
              </Link>
            </Typography>
            {user && (
              <>
                <Button color="inherit" component={Link} to="/dashboard">
                  Dashboard
                </Button>
                <Button color="inherit" component={Link} to="/strategies">
                  Strategies
                </Button>
                <Button color="inherit" component={Link} to="/my-bots">
                  My Bots
                </Button>
                <Button color="inherit" component={Link} to="/portfolio">
                  Portfolio
                </Button>
                <Button color="inherit" component={Link} to="/account">
                  Account
                </Button>
                <Button color="inherit" onClick={logout}>
                  Logout
                </Button>
              </>
            )}
            {!user && (
              <Button color="inherit" component={Link} to="/login">
                Login
              </Button>
            )}
            <Button onClick={toggleTheme} color="inherit">
              {isDarkMode ? 'Light Mode' : 'Dark Mode'}
            </Button>
          </Toolbar>
        </AppBar>
        <Box component="main" sx={{ p: 3 }}>
          <Routes>
            <Route path="/" element={user ? <Navigate to="/dashboard" /> : <Navigate to="/login" />} />
            <Route path="/login" element={!user ? <Login /> : <Navigate to="/dashboard" />} />
            <Route path="/dashboard" element={user ? <Dashboard /> : <Navigate to="/login" />} />
            <Route path="/strategies" element={user ? <Strategies /> : <Navigate to="/login" />} />
            <Route path="/my-bots" element={user ? <MyBots /> : <Navigate to="/login" />} />
            <Route path="/portfolio" element={user ? <Portfolio /> : <Navigate to="/login" />} />
            <Route path="/account" element={user ? <AccountPage /> : <Navigate to="/login" />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

function App() {
  const [googleClientId, setGoogleClientId] = useState(null);
  const [loadingClientId, setLoadingClientId] = useState(true);
  const [errorClientId, setErrorClientId] = useState(null);

  useEffect(() => {
    const fetchGoogleClientId = async () => {
      try {
        const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
        const response = await axios.get(`${apiUrl}/api/config/google-client-id`);
        if (response.data.clientId) {
          setGoogleClientId(response.data.clientId);
        } else {
          throw new Error('Client ID not found in response');
        }
      } catch (error) {
        console.error("Fatal Error: Could not fetch Google Client ID from backend.", error);
        setErrorClientId("Could not load configuration from server.");
      } finally {
        setLoadingClientId(false);
      }
    };

    fetchGoogleClientId();
  }, []);

  if (loadingClientId) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading configuration...</Typography>
      </Box>
    );
  }

  if (errorClientId || !googleClientId) {
    return (
       <Box display="flex" justifyContent="center" alignItems="center" height="100vh" flexDirection="column">
         <Typography color="error" variant="h6">Error</Typography>
         <Typography color="error">{errorClientId || "Google Client ID could not be loaded."}</Typography>
       </Box>
     );
  }

  return (
    <GoogleOAuthProvider clientId={googleClientId}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </GoogleOAuthProvider>
  );
}

export default App;
