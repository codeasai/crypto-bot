import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Pagination,
  CircularProgress,
  Alert,
  Chip
} from '@mui/material';
import axios from 'axios';

const Portfolio = () => {
  const [portfolio, setPortfolio] = useState(null);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const itemsPerPage = 20;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/portfolio');
        // กรองเฉพาะเหรียญที่มีจำนวนมากกว่า 0
        const filteredAssets = response.data.assets.filter(asset => asset.amount > 0);
        setPortfolio({
          ...response.data,
          assets: filteredAssets
        });
        setError(null);
      } catch (err) {
        setError('เกิดข้อผิดพลาดในการดึงข้อมูล');
        console.error(err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!portfolio) {
    return <CircularProgress />;
  }

  // คำนวณจำนวนหน้าทั้งหมด
  const totalPages = Math.ceil(portfolio.assets.length / itemsPerPage);
  
  // คำนวณข้อมูลที่จะแสดงในหน้าปัจจุบัน
  const startIndex = (page - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentAssets = portfolio.assets.slice(startIndex, endIndex);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            พอร์ตโฟลิโอ
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Typography variant="h6">
              มูลค่ารวม: ${portfolio.totalValue.toFixed(2)}
            </Typography>
            <Chip 
              label={`${portfolio.assets.length} เหรียญ`} 
              color="primary" 
              variant="outlined"
            />
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>เหรียญ</TableCell>
                  <TableCell align="right">จำนวน</TableCell>
                  <TableCell align="right">มูลค่า (USD)</TableCell>
                  <TableCell align="right">% ของพอร์ต</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {currentAssets.map((asset) => (
                  <TableRow key={asset.symbol}>
                    <TableCell component="th" scope="row">
                      {asset.symbol}
                    </TableCell>
                    <TableCell align="right">
                      {asset.amount.toFixed(8)}
                    </TableCell>
                    <TableCell align="right">
                      ${asset.value.toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      {((asset.value / portfolio.totalValue) * 100).toFixed(2)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Pagination
              count={totalPages}
              page={page}
              onChange={handlePageChange}
              color="primary"
            />
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Portfolio; 