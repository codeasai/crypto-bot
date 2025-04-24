import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Container,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  CircularProgress,
  Snackbar,
  FormControlLabel,
  Switch
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Code as CodeIcon,
  RocketLaunch as CreateBotIcon
} from '@mui/icons-material';
import { getStrategies, createBot } from '../services/api';
import { useNavigate } from 'react-router-dom';

const Strategies = () => {
  const navigate = useNavigate();
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [newBotFormData, setNewBotFormData] = useState({
    id: '',
    config: {
        api_key: '',
        api_secret: '',
        is_testnet: true,
        symbols: ['BTC/USDT'],
        timeframe: '5m',
        strategy: '',
        strategy_params: {},
        max_position_percentage: 0.20,
        max_trade_percentage: 0.05,
        check_interval: 60,
        maturity_level: 1
    }
  });
  const [createLoading, setCreateLoading] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  useEffect(() => {
    fetchStrategies();
  }, []);

  const fetchStrategies = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getStrategies();
      setStrategies(data || []);
    } catch (err) {
      setError(err.message || 'Could not fetch strategies');
      console.error('Error fetching strategies:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = (strategy) => {
    setSelectedStrategy(strategy);
    const initialParams = {};
    strategy.parameters?.forEach(param => {
        initialParams[param.name] = param.default ?? '';
    });
    setNewBotFormData({
        id: `${strategy.id}_${Date.now().toString().slice(-5)}`,
        config: {
            api_key: '',
            api_secret: '',
            is_testnet: true,
            symbols: ['BTC/USDT'],
            timeframe: '5m',
            strategy: strategy.id,
            strategy_params: initialParams,
            max_position_percentage: 0.20,
            max_trade_percentage: 0.05,
            check_interval: 60,
            maturity_level: 1
        }
    });
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedStrategy(null);
    setError(null);
  };

  const handleFormChange = (path, value) => {
    setNewBotFormData(prev => {
        const keys = path.split('.');
        let current = prev;
        for (let i = 0; i < keys.length - 1; i++) {
            current = current[keys[i]];
            if (current === undefined) return prev;
        }
        current[keys[keys.length - 1]] = value;
        return { ...prev };
    });
  };

  const handleCreateBot = async () => {
    setCreateLoading(true);
    setError(null);
    try {
        if (!newBotFormData.id || !newBotFormData.config.strategy) {
            throw new Error("Bot ID and Strategy are required.");
        }

      const response = await createBot(newBotFormData);
      if (response.status === 'success') {
        setSnackbarMessage(`Bot '${newBotFormData.id}' created successfully!`);
        setSnackbarOpen(true);
        handleCloseDialog();
        navigate('/my-bots');
      } else {
        throw new Error(response.message || 'Failed to create bot');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while creating the bot.');
      console.error('Error creating bot:', err);
    } finally {
      setCreateLoading(false);
    }
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Available Trading Strategies
        </Typography>

        {error && !loading && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {strategies.length === 0 && !error && !loading ? (
          <Alert severity="info">
            No strategies available.
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {strategies.map((strategy) => (
              <Grid item xs={12} sm={6} md={4} key={strategy.id}>
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {strategy.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {strategy.description}
                    </Typography>
                    {strategy.parameters && strategy.parameters.length > 0 && (
                      <Box mt={2}>
                        <Typography variant="subtitle2" gutterBottom>
                          Parameters:
                        </Typography>
                        <List dense sx={{ maxHeight: 150, overflow: 'auto' }}>
                          {strategy.parameters.map((param) => (
                            <ListItem key={param.name} disablePadding>
                              <ListItemText
                                primary={param.label || param.name}
                                secondary={`(${param.type || 'text'}) ${param.min !== undefined ? `Min: ${param.min}` : ''} ${param.max !== undefined ? `Max: ${param.max}` : ''} ${param.default !== undefined ? `Default: ${param.default}` : ''}`}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}
                  </CardContent>
                  <CardActions sx={{ justifyContent: 'flex-end' }}>
                    <Button
                      size="small"
                      variant="contained"
                      color="primary"
                      startIcon={<CreateBotIcon />}
                      onClick={() => handleOpenDialog(strategy)}
                    >
                      Create Bot
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Create Bot from: {selectedStrategy?.name}</DialogTitle>
        <DialogContent>
             {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
             <TextField
                 autoFocus
                 margin="dense"
                 label="Bot ID" required
                 fullWidth
                 value={newBotFormData.id}
                 onChange={(e) => handleFormChange('id', e.target.value)}
                 helperText="Unique identifier for your bot"
             />
            <TextField
                margin="dense"
                label="API Key (Binance)" required
                fullWidth
                value={newBotFormData.config.api_key}
                onChange={(e) => handleFormChange('config.api_key', e.target.value)}
             />
             <TextField
                 margin="dense"
                 label="API Secret (Binance)" required
                 type="password"
                 fullWidth
                 value={newBotFormData.config.api_secret}
                 onChange={(e) => handleFormChange('config.api_secret', e.target.value)}
             />
             <FormControl fullWidth margin="dense">
                <InputLabel>Symbols (e.g., BTC/USDT,ETH/USDT)</InputLabel>
                <Select
                    multiple required
                    value={newBotFormData.config.symbols}
                    onChange={(e) => handleFormChange('config.symbols', typeof e.target.value === 'string' ? e.target.value.split(',').map(s=>s.trim()).filter(s=>s) : e.target.value)}
                    renderValue={(selected) => selected.join(', ')}
                    label="Symbols (e.g., BTC/USDT,ETH/USDT)" >
                      <MenuItem value="BTC/USDT">BTC/USDT</MenuItem>
                      <MenuItem value="ETH/USDT">ETH/USDT</MenuItem>
                      <MenuItem value="BNB/USDT">BNB/USDT</MenuItem>
                      <MenuItem value="XRP/USDT">XRP/USDT</MenuItem>
                      <MenuItem value="ADA/USDT">ADA/USDT</MenuItem>
                      <MenuItem value="SOL/USDT">SOL/USDT</MenuItem>
                </Select>
             </FormControl>

             <Typography sx={{ mt: 2, mb: 1 }}>Strategy Parameters ({selectedStrategy?.name}):</Typography>
             {selectedStrategy?.parameters?.map(param => (
                 <TextField
                    key={param.name}
                    margin="dense"
                    label={param.label || param.name} required
                    type={param.type === 'number' ? 'number' : 'text'}
                    fullWidth
                    value={newBotFormData.config.strategy_params[param.name] ?? ''}
                    onChange={(e) => handleFormChange(`config.strategy_params.${param.name}`, param.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
                    helperText={`Min: ${param.min ?? 'N/A'}, Max: ${param.max ?? 'N/A'}, Default: ${param.default ?? 'N/A'}`}
                    InputLabelProps={{ shrink: true }}
                    inputProps={{
                        min: param.min,
                        max: param.max,
                        step: param.type === 'number' ? 'any' : undefined
                    }}
                 />
             ))}

            <Typography sx={{ mt: 2, mb: 1 }}>Advanced Settings:</Typography>
             <Grid container spacing={2}>
                <Grid item xs={6}>
                 <TextField
                    margin="dense"
                    label="Max Position %" type="number"
                    fullWidth InputLabelProps={{ shrink: true }}
                    value={newBotFormData.config.max_position_percentage * 100}
                    onChange={(e) => handleFormChange('config.max_position_percentage', parseFloat(e.target.value) / 100)}
                 />
                </Grid>
                <Grid item xs={6}>
                 <TextField
                    margin="dense"
                    label="Max Trade %" type="number"
                    fullWidth InputLabelProps={{ shrink: true }}
                    value={newBotFormData.config.max_trade_percentage * 100}
                    onChange={(e) => handleFormChange('config.max_trade_percentage', parseFloat(e.target.value) / 100)}
                 />
                </Grid>
                 <Grid item xs={6}>
                 <TextField
                    margin="dense"
                    label="Check Interval (sec)" type="number"
                    fullWidth InputLabelProps={{ shrink: true }}
                    value={newBotFormData.config.check_interval}
                    onChange={(e) => handleFormChange('config.check_interval', parseInt(e.target.value, 10) || 60)}
                 />
                </Grid>
                 <Grid item xs={6}>
                    <FormControl fullWidth margin="dense">
                        <InputLabel>Maturity Level</InputLabel>
                         <Select
                            label="Maturity Level"
                            value={newBotFormData.config.maturity_level}
                            onChange={(e) => handleFormChange('config.maturity_level', parseInt(e.target.value, 10))}
                         >
                            <MenuItem value={1}>1 - Analysis Only</MenuItem>
                            <MenuItem value={2}>2 - Auto Sell Profit</MenuItem>
                            <MenuItem value={3}>3 - Full Auto Trade</MenuItem>
                         </Select>
                   </FormControl>
                 </Grid>
                 <Grid item xs={12}>
                     <FormControlLabel
                        control={<Switch checked={newBotFormData.config.is_testnet} onChange={(e) => handleFormChange('config.is_testnet', e.target.checked)} />}
                        label="Use Binance Testnet"
                     />
                 </Grid>
             </Grid>

        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button
            onClick={handleCreateBot}
            variant="contained"
            color="primary"
            disabled={createLoading}
            startIcon={createLoading ? <CircularProgress size={20} /> : null}
           >
            Create Bot
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
      />

    </Container>
  );
};

export default Strategies; 