import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert
} from '@mui/material';
import { createBot, getStrategies } from '../api';

const StrategyList = () => {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [formData, setFormData] = useState({});
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const data = await getStrategies();
        setStrategies(data);
        setError('');
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchStrategies();
  }, []);

  const handleStrategySelect = (strategy) => {
    setSelectedStrategy(strategy);
    setFormData({});
    setOpenDialog(true);
  };

  const handleClose = () => {
    setOpenDialog(false);
    setSelectedStrategy(null);
    setFormData({});
    setError(null);
    setSuccess(false);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async () => {
    try {
      setError(null);
      await createBot({
        strategy_id: selectedStrategy.id,
        parameters: formData
      });
      setSuccess(true);
      setTimeout(handleClose, 2000);
    } catch (err) {
      setError('Failed to create bot. Please try again.');
    }
  };

  if (loading) return <div>กำลังโหลด...</div>;
  if (error) return <div className="error">{error}</div>;
  if (strategies.length === 0) return <div>ไม่พบกลยุทธ์ที่บันทึกไว้</div>;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Available Strategies
      </Typography>
      
      <Grid container spacing={3}>
        {strategies.map((strategy) => (
          <Grid item xs={12} md={4} key={strategy.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {strategy.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {strategy.description}
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={() => handleStrategySelect(strategy)}
                >
                  Create Bot
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={openDialog} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Bot</DialogTitle>
        <DialogContent>
          {selectedStrategy && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                {selectedStrategy.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                {selectedStrategy.description}
              </Typography>
              
              {selectedStrategy.parameters.map((param) => (
                <TextField
                  key={param.name}
                  name={param.name}
                  label={param.label}
                  type={param.type}
                  fullWidth
                  margin="normal"
                  value={formData[param.name] || ''}
                  onChange={handleInputChange}
                  inputProps={{
                    min: param.min,
                    max: param.max
                  }}
                />
              ))}

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}

              {success && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Bot created successfully!
                </Alert>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
          <Button 
            onClick={handleSubmit}
            variant="contained"
            disabled={success}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default StrategyList; 