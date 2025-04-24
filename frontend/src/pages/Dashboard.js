import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  CircularProgress,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

const Dashboard = () => {
  const [botStatus, setBotStatus] = useState(null);
  const [priceData, setPriceData] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [orders, setOrders] = useState([]);
  const [error, setError] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusRes, priceRes, portfolioRes, ordersRes] = await Promise.all([
          axios.get('http://localhost:5000/api/status'),
          axios.get(`http://localhost:5000/api/price?symbol=${selectedSymbol}`),
          axios.get('http://localhost:5000/api/portfolio'),
          axios.get(`http://localhost:5000/api/orders?symbol=${selectedSymbol}`)
        ]);

        setBotStatus(statusRes.data);
        setPriceData(priceRes.data);
        setPortfolio(portfolioRes.data);
        setOrders(ordersRes.data);
        setError(null);
      } catch (err) {
        setError('เกิดข้อผิดพลาดในการดึงข้อมูล');
        console.error(err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const handleSymbolChange = (event) => {
    setSelectedSymbol(event.target.value);
  };

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!botStatus) {
    return <CircularProgress />;
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {/* เลือกเหรียญ */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <FormControl fullWidth>
                <InputLabel>เลือกเหรียญ</InputLabel>
                <Select
                  value={selectedSymbol}
                  onChange={handleSymbolChange}
                  label="เลือกเหรียญ"
                >
                  {botStatus.config?.symbols?.map((symbol) => (
                    <MenuItem key={symbol} value={symbol}>
                      {symbol}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        {/* สถานะบอท */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">สถานะบอท - {selectedSymbol}</Typography>
              <Typography>อัพเดทล่าสุด: {botStatus.last_update}</Typography>
              <Typography>สัญญาณ: {botStatus.signal || 'ไม่มี'}</Typography>
              <Typography>EMA Short: {botStatus.ema_data?.ema_short?.toFixed(2)}</Typography>
              <Typography>EMA Long: {botStatus.ema_data?.ema_long?.toFixed(2)}</Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* พอร์ตโฟลิโอ */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">พอร์ตโฟลิโอ - {selectedSymbol}</Typography>
              <Typography>มูลค่ารวม: ${portfolio?.totalValue?.toFixed(2)}</Typography>
              {portfolio?.assets?.filter(asset => asset.symbol === selectedSymbol.split('/')[0]).map((asset) => (
                <Typography key={asset.symbol}>
                  {asset.symbol}: {asset.amount.toFixed(8)} (${asset.value.toFixed(2)})
                </Typography>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* กราฟราคา */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6">กราฟราคา - {selectedSymbol}</Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="price" stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* ออเดอร์ที่เปิดอยู่ */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6">ประวัติออเดอร์ - {selectedSymbol}</Typography>
              {orders.map((order) => (
                <Typography key={order.id} sx={{ 
                  color: order.is_open ? 'success.main' : 'text.secondary',
                  display: 'flex',
                  justifyContent: 'space-between',
                  mb: 1
                }}>
                  <span>
                    {order.symbol} - {order.type}: {order.amount} @ {order.price}
                  </span>
                  <span>
                    {order.is_open ? 'เปิดอยู่' : 'ปิดแล้ว'}
                  </span>
                </Typography>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 