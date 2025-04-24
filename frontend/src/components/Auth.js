import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';

const Auth = () => {
  const { login } = useAuth();

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      const decoded = jwtDecode(credentialResponse.credential);
      await login({
        sub: decoded.sub,
        email: decoded.email,
        name: decoded.name,
        picture: decoded.picture
      });
    } catch (error) {
      console.error('Error decoding Google token:', error);
    }
  };

  return (
    <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
      <GoogleLogin
        onSuccess={handleGoogleSuccess}
        onError={() => console.error('Login Failed')}
        useOneTap
      />
    </GoogleOAuthProvider>
  );
};

export default Auth; 