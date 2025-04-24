import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import {
  getStatus,
  createBot,
  deleteBot,
  startBot,
  stopBot,
  updateBotConfig
} from '../services/api';

const BotControl = () => {
  const { t } = useTranslation();
  const [bots, setBots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedBot, setSelectedBot] = useState(null);
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);
  const [newBotConfig, setNewBotConfig] = useState({
    id: '',
    config: {
        api_key: '',
        api_secret: '',
        is_testnet: true,
        symbols: ['BTC/USDT'],
        timeframe: '5m',
        strategy: 'cost_basis',
        strategy_params: {
            profit_target_percentage: 0.05,
            buy_dip_percentage: 0.10
        },
        max_position_percentage: 0.20,
        max_trade_percentage: 0.05,
        check_interval: 60,
        maturity_level: 1
    }
  });

  useEffect(() => {
    fetchBots();
  }, []);

  const fetchBots = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getStatus();
      setBots(data.bots || []);
    } catch (err) {
      setError(err.message || t('error'));
    } finally {
      setLoading(false);
    }
  };

  const clearMessages = () => {
    setTimeout(() => {
        setError(null);
        setSuccess(null);
    }, 3000);
  };

  const handleCreateBot = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      await createBot(newBotConfig);
      setSuccess(t('success'));
      setOpenCreateDialog(false);
      setNewBotConfig({
        id: '',
        config: {
            api_key: '', api_secret: '', is_testnet: true, symbols: ['BTC/USDT'],
            timeframe: '5m', strategy: 'cost_basis', strategy_params: { profit_target_percentage: 0.05, buy_dip_percentage: 0.10 },
            max_position_percentage: 0.20, max_trade_percentage: 0.05, check_interval: 60, maturity_level: 1
        }
      });
      fetchBots();
      clearMessages();
    } catch (err) {
      setError(err.message || t('error'));
      clearMessages();
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteBot = async (botId) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      await deleteBot(botId);
      setSuccess(t('success'));
      fetchBots();
      setSelectedBot(null);
      clearMessages();
    } catch (err) {
      setError(err.message || t('error'));
      clearMessages();
    } finally {
      setLoading(false);
    }
  };

  const handleStartBot = async (botId) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      await startBot(botId);
      setSuccess(t('success'));
      fetchBots();
      clearMessages();
    } catch (err) {
      setError(err.message || t('error'));
      clearMessages();
    } finally {
      setLoading(false);
    }
  };

  const handleStopBot = async (botId) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      await stopBot(botId);
      setSuccess(t('success'));
      fetchBots();
      clearMessages();
    } catch (err) {
      setError(err.message || t('error'));
      clearMessages();
    } finally {
      setLoading(false);
    }
  };

  const handleSelectBot = (bot) => {
    setSelectedBot(bot);
    setEditingConfig(JSON.parse(JSON.stringify(bot.config)));
  };

  const handleEditingConfigChange = (field, value) => {
    setEditingConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleStrategyParamsChange = (param, value) => {
    setEditingConfig(prev => ({
        ...prev,
        strategy_params: {
            ...prev.strategy_params,
            [param]: parseFloat(value)
        }
    }));
  };

  const handleSaveConfig = async (botId) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      await updateBotConfig(botId, editingConfig);
      setSuccess(t('success'));
      fetchBots();
      setSelectedBot(null);
      setEditingConfig(null);
      clearMessages();
    } catch (err) {
      setError(err.message || t('error'));
      clearMessages();
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleNewBotConfigChange = (field, value) => {
    setNewBotConfig(prev => ({
        ...prev,
        config: {
            ...prev.config,
            [field]: value
        }
    }));
  };

  const handleNewBotStrategyParamsChange = (param, value) => {
     setNewBotConfig(prev => ({
        ...prev,
        config: {
            ...prev.config,
            strategy_params: {
                ...prev.config.strategy_params,
                [param]: parseFloat(value)
            }
        }
    }));
  };

  if (loading && !bots.length) {
    return <CircularProgress />;
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">{t('my_bots')}</Typography>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => setOpenCreateDialog(true)}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                  disabled={loading}
                >
                  {t('create_bot')}
                </Button>
              </Box>

              {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
              {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>{t('bot_id')}</TableCell>
                      <TableCell>{t('status')}</TableCell>
                      <TableCell>{t('strategy')}</TableCell>
                      <TableCell>{t('symbols')}</TableCell>
                      <TableCell>{t('settings')}</TableCell>
                      <TableCell align="right">{t('actions')}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {bots.map((bot) => (
                      <TableRow key={bot.id}>
                        <TableCell>{bot.id}</TableCell>
                        <TableCell>
                          {bot.is_running ? (
                            <Typography color="success.main">{t('running')}</Typography>
                          ) : (
                            <Typography color="error.main">{t('stopped')}</Typography>
                          )}
                        </TableCell>
                        <TableCell>{bot.config?.strategy || 'N/A'}</TableCell>
                        <TableCell>{bot.config?.symbols?.join(', ') || 'N/A'}</TableCell>
                        <TableCell>
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => handleSelectBot(bot)}
                            disabled={loading}
                          >
                            {t('settings')}
                          </Button>
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                            {bot.is_running ? (
                              <Button
                                variant="contained"
                                color="warning"
                                size="small"
                                onClick={() => handleStopBot(bot.id)}
                                disabled={loading}
                              >
                                {t('stop')}
                              </Button>
                            ) : (
                              <Button
                                variant="contained"
                                color="success"
                                size="small"
                                onClick={() => handleStartBot(bot.id)}
                                disabled={loading}
                              >
                                {t('start')}
                              </Button>
                            )}
                            <Button
                              variant="outlined"
                              color="error"
                              size="small"
                              onClick={() => handleDeleteBot(bot.id)}
                              disabled={loading}
                            >
                              {t('delete')}
                            </Button>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {selectedBot && editingConfig && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {t('settings')} {selectedBot.id}
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>{t('strategy')}</InputLabel>
                      <Select
                        value={editingConfig.strategy || ''}
                        onChange={(e) => handleEditingConfigChange('strategy', e.target.value)}
                      >
                        <MenuItem value="cost_basis">Cost Basis</MenuItem>
                        <MenuItem value="moving_average">Moving Average</MenuItem>
                        <MenuItem value="rsi">RSI</MenuItem>
                        <MenuItem value="ema_crossover">EMA Crossover</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  {editingConfig.strategy === 'cost_basis' && (
                    <>
                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth label={t('profit_target')}
                          type="number" InputLabelProps={{ shrink: true }}
                          value={(editingConfig.strategy_params?.profit_target_percentage || 0) * 100}
                          onChange={(e) => handleStrategyParamsChange('profit_target_percentage', e.target.value / 100)}
                        />
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth label={t('buy_dip')}
                          type="number" InputLabelProps={{ shrink: true }}
                          value={(editingConfig.strategy_params?.buy_dip_percentage || 0) * 100}
                          onChange={(e) => handleStrategyParamsChange('buy_dip_percentage', e.target.value / 100)}
                        />
                      </Grid>
                    </>
                  )}
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth label={t('max_position')}
                      type="number" InputLabelProps={{ shrink: true }}
                      value={(editingConfig.max_position_percentage || 0) * 100}
                      onChange={(e) => handleEditingConfigChange('max_position_percentage', parseFloat(e.target.value) / 100)}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth label={t('max_trade')}
                      type="number" InputLabelProps={{ shrink: true }}
                      value={(editingConfig.max_trade_percentage || 0) * 100}
                      onChange={(e) => handleEditingConfigChange('max_trade_percentage', parseFloat(e.target.value) / 100)}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                     <TextField
                        fullWidth label={t('symbols')}
                        helperText="Comma-separated, e.g., BTC/USDT,ETH/USDT"
                        value={editingConfig.symbols?.join(',') || ''}
                        onChange={(e) => handleEditingConfigChange('symbols', e.target.value.split(',').map(s => s.trim()).filter(s => s))}
                     />
                  </Grid>
                   <Grid item xs={12} md={6}>
                     <TextField
                        fullWidth label={t('check_interval')}
                        type="number" InputLabelProps={{ shrink: true }}
                        value={editingConfig.check_interval || 60}
                        onChange={(e) => handleEditingConfigChange('check_interval', parseInt(e.target.value, 10))}
                     />
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={() => handleSaveConfig(selectedBot.id)}
                      disabled={loading}
                      sx={{ mr: 1 }}
                    >
                      {t('save')}
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => {setSelectedBot(null); setEditingConfig(null);}}
                      disabled={loading}
                    >
                      {t('cancel')}
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

      </Grid>

      <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{t('create_bot')}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth required
                label={t('bot_id')}
                value={newBotConfig.id}
                onChange={(e) => setNewBotConfig(prev => ({ ...prev, id: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth required
                label={t('api_key')}
                value={newBotConfig.config.api_key}
                onChange={(e) => handleNewBotConfigChange('api_key', e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth required
                label={t('api_secret')}
                type="password"
                value={newBotConfig.config.api_secret}
                onChange={(e) => handleNewBotConfigChange('api_secret', e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>{t('strategy')}</InputLabel>
                <Select
                  value={newBotConfig.config.strategy}
                  onChange={(e) => handleNewBotConfigChange('strategy', e.target.value)}
                >
                  <MenuItem value="cost_basis">Cost Basis</MenuItem>
                  <MenuItem value="moving_average">Moving Average</MenuItem>
                  <MenuItem value="rsi">RSI</MenuItem>
                  <MenuItem value="ema_crossover">EMA Crossover</MenuItem>
                </Select>
              </FormControl>
            </Grid>
             {newBotConfig.config.strategy === 'cost_basis' && (
                 <>
                    <Grid item xs={6}>
                        <TextField
                            fullWidth label={`${t('profit_target')} (%)`}
                            type="number"
                            value={newBotConfig.config.strategy_params.profit_target_percentage * 100}
                            onChange={(e) => handleNewBotStrategyParamsChange('profit_target_percentage', e.target.value / 100)}
                         />
                    </Grid>
                    <Grid item xs={6}>
                        <TextField
                            fullWidth label={`${t('buy_dip')} (%)`}
                            type="number"
                            value={newBotConfig.config.strategy_params.buy_dip_percentage * 100}
                            onChange={(e) => handleNewBotStrategyParamsChange('buy_dip_percentage', e.target.value / 100)}
                        />
                    </Grid>
                 </>
             )}
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>{t('symbols')}</InputLabel>
                <Select
                  multiple required
                  value={newBotConfig.config.symbols}
                  onChange={(e) => handleNewBotConfigChange('symbols', typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                  renderValue={(selected) => selected.join(', ')}
                >
                  <MenuItem value="BTC/USDT">BTC/USDT</MenuItem>
                  <MenuItem value="ETH/USDT">ETH/USDT</MenuItem>
                  <MenuItem value="BNB/USDT">BNB/USDT</MenuItem>
                  <MenuItem value="XRP/USDT">XRP/USDT</MenuItem>
                  <MenuItem value="ADA/USDT">ADA/USDT</MenuItem>
                </Select>
              </FormControl>
            </Grid>
             <Grid item xs={6}>
                 <TextField
                    fullWidth label={`${t('max_position')} (%)`}
                    type="number"
                    value={newBotConfig.config.max_position_percentage * 100}
                    onChange={(e) => handleNewBotConfigChange('max_position_percentage', parseFloat(e.target.value) / 100)}
                 />
             </Grid>
            <Grid item xs={6}>
                 <TextField
                    fullWidth label={`${t('max_trade')} (%)`}
                    type="number"
                    value={newBotConfig.config.max_trade_percentage * 100}
                    onChange={(e) => handleNewBotConfigChange('max_trade_percentage', parseFloat(e.target.value) / 100)}
                 />
            </Grid>
            <Grid item xs={6}>
                 <TextField
                    fullWidth label={`${t('check_interval')} (sec)`}
                    type="number"
                    value={newBotConfig.config.check_interval}
                    onChange={(e) => handleNewBotConfigChange('check_interval', parseInt(e.target.value, 10))}
                 />
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>{t('maturity_level')}</InputLabel>
                 <Select
                    value={newBotConfig.config.maturity_level}
                    onChange={(e) => handleNewBotConfigChange('maturity_level', e.target.value)}
                 >
                    <MenuItem value={1}>1 - Analysis Only</MenuItem>
                    <MenuItem value={2}>2 - Auto Sell Profit</MenuItem>
                    <MenuItem value={3}>3 - Full Auto Trade</MenuItem>
                 </Select>
               </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCreateDialog(false)}>{t('cancel')}</Button>
          <Button onClick={handleCreateBot} variant="contained" color="primary" disabled={loading}>
            {t('create_bot')}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default BotControl; 