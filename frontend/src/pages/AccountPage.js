import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Avatar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
  Alert,
  MenuItem
} from '@mui/material';
import {
  Person as PersonIcon,
  Email as EmailIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Edit as EditIcon
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import { updateUserProfile } from '../services/api';

const AccountPage = () => {
  const { user, setUser } = useAuth();
  const [editMode, setEditMode] = useState(false);
  const [editedUser, setEditedUser] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    if (user && editMode) {
      setEditedUser(JSON.parse(JSON.stringify(user)));
    } else if (!editMode) {
       setEditedUser(null);
       setError(null);
       setSuccess(null);
    }
  }, [user, editMode]);

  const handleEditClick = () => {
    setEditMode(true);
  };

  const handleSave = async () => {
    setError(null);
    setSuccess(null);
    if (!editedUser) return;

    const updates = {
        name: editedUser.name,
        preferences: editedUser.preferences
    };

    try {
      const response = await updateUserProfile(updates);
      if (response.success) {
        const updatedUserData = { ...user, ...updates };
        setUser(updatedUserData);
        localStorage.setItem('user', JSON.stringify(updatedUserData));
        setSuccess(response.message || 'Profile updated successfully!');
        setEditMode(false);
      } else {
        setError(response.message || 'Failed to update profile.');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while updating profile.');
      console.error('Error updating profile:', err);
    }
  };

  const handleCancel = () => {
    setEditMode(false);
  };

  if (!user) {
    return <Typography>Loading user data...</Typography>;
  }

  return (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          My Account
        </Typography>
        {!editMode && (
            <Button
              variant="outlined"
              startIcon={<EditIcon />}
              onClick={handleEditClick}
            >
              Edit Profile
            </Button>
        )}
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Avatar
              src={user.picture}
              alt={user.name}
              sx={{ width: 120, height: 120, mx: 'auto', mb: 2 }}
            />
            <Typography variant="h5" gutterBottom>
              {user.name}
            </Typography>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              {user.email}
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper>
            <List>
              <ListItem>
                <ListItemIcon>
                  <PersonIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Personal Information"
                  secondary={
                    <>
                      <Typography variant="body2">Name: {user.name}</Typography>
                      <Typography variant="body2">Email: {user.email}</Typography>
                    </>
                  }
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <EmailIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Email Settings"
                  secondary="Notifications and communication preferences"
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <SecurityIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Security"
                  secondary="Password and account security settings"
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <SettingsIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Preferences"
                  secondary={
                    <>
                      <Typography variant="body2">Language: {user.preferences?.language || 'English'}</Typography>
                      <Typography variant="body2">Theme: {user.preferences?.theme || 'Dark'}</Typography>
                    </>
                  }
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      <Dialog open={editMode} onClose={handleCancel} fullWidth maxWidth="xs">
        <DialogTitle>Edit Profile</DialogTitle>
        <DialogContent>
          {editedUser && (
            <Box sx={{ pt: 1 }}>
              <TextField
                fullWidth autoFocus
                label="Name"
                value={editedUser.name || ''}
                onChange={(e) => setEditedUser({ ...editedUser, name: e.target.value })}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Email"
                value={editedUser.email || ''}
                disabled
                margin="normal"
              />
              <TextField
                fullWidth
                label="Language"
                value={editedUser.preferences?.language || 'English'}
                onChange={(e) => setEditedUser({
                  ...editedUser,
                  preferences: { ...(editedUser.preferences || {}), language: e.target.value }
                })}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Theme"
                select
                value={editedUser.preferences?.theme || 'Dark'}
                onChange={(e) => setEditedUser({
                  ...editedUser,
                  preferences: { ...(editedUser.preferences || {}), theme: e.target.value }
                })}
                margin="normal"
              >
                 <MenuItem value="Dark">Dark</MenuItem>
                 <MenuItem value="Light">Light</MenuItem>
              </TextField>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancel}>Cancel</Button>
          <Button onClick={handleSave} variant="contained" color="primary">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AccountPage; 