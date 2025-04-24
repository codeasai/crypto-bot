import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Divider,
  Grid,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer
} from 'recharts';
import { getPortfolioData, getActiveBots } from '../api';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const Account = () => {
  const [portfolioData, setPortfolioData] = useState([]);
  const [activeBots, setActiveBots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [portfolioResponse, botsResponse] = await Promise.all([
          getPortfolioData(),
          getActiveBots()
        ]);
        
        // แปลงข้อมูลพอร์ตโฟลิโอให้อยู่ในรูปแบบที่ recharts ต้องการ
        const formattedPortfolioData = portfolioResponse.map(item => ({
          name: item.symbol,
          value: item.value
        }));
        
        setPortfolioData(formattedPortfolioData);
        setActiveBots(botsResponse);
        setError(null);
      } catch (err) {
        setError('Failed to fetch data. Please try again later.');
        console.error('Error fetching data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // อัพเดทข้อมูลทุก 30 วินาที
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Grid container spacing={3}>
        {/* Portfolio Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={portfolioData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {portfolioData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Active Bots */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Bots
              </Typography>
              {activeBots.length === 0 ? (
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  alignItems: 'center',
                  height: 200,
                  color: 'text.secondary'
                }}>
                  <Typography variant="body1">
                    No active bots at the moment
                  </Typography>
                </Box>
              ) : (
                <List>
                  {activeBots.map((bot, index) => (
                    <React.Fragment key={bot.id}>
                      <ListItem>
                        <ListItemAvatar>
                          <Avatar sx={{ bgcolor: bot.status === 'Running' ? 'success.main' : 'warning.main' }}>
                            {bot.status === 'Running' ? 'R' : 'P'}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={bot.name}
                          secondary={
                            <>
                              <Typography component="span" variant="body2" color={bot.profit.startsWith('+') ? 'success.main' : 'error.main'}>
                                {bot.profit}
                              </Typography>
                              {` • Last update: ${bot.lastUpdate}`}
                            </>
                          }
                        />
                      </ListItem>
                      {index < activeBots.length - 1 && <Divider variant="inset" component="li" />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Account; 