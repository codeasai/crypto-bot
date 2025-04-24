import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  Chip // เพิ่ม Chip สำหรับแสดงสถานะ
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon // เพิ่มไอคอน Refresh
} from '@mui/icons-material';
import { getStatus, startBot, stopBot, deleteBot } from '../services/api';
import { Link } from 'react-router-dom'; // สำหรับ Link ไปหน้า Config (ถ้ามี)

const MyBots = () => {
  const [bots, setBots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionLoading, setActionLoading] = useState({}); // State สำหรับ loading ของแต่ละปุ่ม

  // ใช้ useCallback เพื่อไม่ให้ fetchBots ถูกสร้างใหม่ทุกครั้งที่ render
  const fetchBots = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getStatus();
      setBots(data.bots || []);
    } catch (err) {
      setError(err.message || 'Could not fetch bot status');
      console.error('Error fetching bot status:', err);
    } finally {
      setLoading(false);
    }
  }, []); // Dependencies array ว่างเปล่า เพราะไม่มี props หรือ state ที่ fetchBots ขึ้นอยู่กับ

  useEffect(() => {
    fetchBots();
    // Optional: ตั้ง Interval สำหรับ refresh ข้อมูล
    // const intervalId = setInterval(fetchBots, 30000); // Refresh ทุก 30 วินาที
    // return () => clearInterval(intervalId); // Clear interval ตอน unmount
  }, [fetchBots]); // เรียก fetchBots เมื่อ component mount หรือ function fetchBots เปลี่ยน (ซึ่งไม่ควรเปลี่ยน)

  const handleAction = async (actionFunc, botId, actionName) => {
    setActionLoading(prev => ({ ...prev, [botId + actionName]: true }));
    setError(null);
    try {
      const response = await actionFunc(botId);
      if (response.status !== 'success') {
        throw new Error(response.message || `Failed to ${actionName} bot`);
      }
      await fetchBots(); // Refresh list หลังจาก action สำเร็จ
    } catch (err) {
      setError(err.message || `An error occurred while ${actionName}ing the bot.`);
      console.error(`Error ${actionName}ing bot ${botId}:`, err);
    }
     finally {
       setActionLoading(prev => ({ ...prev, [botId + actionName]: false }));
       // อาจจะเพิ่ม Snackbar แจ้งเตือน
    }
  };

  if (loading && bots.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h4" component="h1" gutterBottom>
              My Bots
            </Typography>
            <Tooltip title="Refresh List">
                 <IconButton onClick={fetchBots} disabled={loading}>
                     <RefreshIcon />
                 </IconButton>
             </Tooltip>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Bot ID</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Strategy</TableCell>
                <TableCell>Symbols</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {bots.length === 0 && !loading ? (
                <TableRow>
                    <TableCell colSpan={5} align="center">
                        No bots found. <Link to="/strategies">Create one from Strategies</Link>.
                    </TableCell>
                </TableRow>
              ) : (
                bots.map((bot) => (
                  <TableRow key={bot.id}>
                    <TableCell>{bot.id}</TableCell>
                    <TableCell>
                      {bot.is_running ? (
                        <Chip label="Running" color="success" size="small" />
                      ) : (
                        <Chip label="Stopped" color="error" size="small" />
                      )}
                    </TableCell>
                    <TableCell>{bot.config?.strategy || 'N/A'}</TableCell>
                    <TableCell>{bot.config?.symbols?.join(', ') || 'N/A'}</TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                        {bot.is_running ? (
                          <Button
                            variant="contained"
                            color="warning"
                            size="small"
                            startIcon={actionLoading[bot.id + 'stop'] ? <CircularProgress size={14} color="inherit" /> : <StopIcon />}
                            onClick={() => handleAction(stopBot, bot.id, 'stop')}
                            disabled={actionLoading[bot.id + 'stop'] || loading}
                            sx={{ minWidth: 90 }}
                          >
                            Stop
                          </Button>
                        ) : (
                          <Button
                            variant="contained"
                            color="success"
                            size="small"
                            startIcon={actionLoading[bot.id + 'start'] ? <CircularProgress size={14} color="inherit" /> : <PlayIcon />}
                            onClick={() => handleAction(startBot, bot.id, 'start')}
                            disabled={actionLoading[bot.id + 'start'] || loading}
                             sx={{ minWidth: 90 }}
                          >
                            Start
                          </Button>
                        )}
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          startIcon={actionLoading[bot.id + 'delete'] ? <CircularProgress size={14} color="inherit" /> : <DeleteIcon />}
                          onClick={() => {
                              if (window.confirm(`Are you sure you want to delete bot '${bot.id}'? This action cannot be undone.`)) {
                                  handleAction(deleteBot, bot.id, 'delete');
                              }
                          }}
                          disabled={actionLoading[bot.id + 'delete'] || loading}
                           sx={{ minWidth: 90 }}
                        >
                          Delete
                        </Button>
                        {/* อาจจะเพิ่มปุ่ม Link ไปหน้า Config ของ Bot */}
                        {/* <Button component={Link} to={`/bots/${bot.id}/config`} size="small">Config</Button> */}
                      </Box>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
         {loading && bots.length > 0 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                <CircularProgress size={24} />
                <Typography sx={{ ml: 1 }}>Refreshing...</Typography>
            </Box>
         )}
      </Paper>
    </Box>
  );
};

export default MyBots; 