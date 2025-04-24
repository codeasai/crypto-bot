import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Box
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  AccountBalance as PortfolioIcon,
  Settings as SettingsIcon,
  Robot as BotIcon
} from '@mui/icons-material';

const Sidebar = () => {
  const location = useLocation();

  const menuItems = [
    { text: 'แดชบอร์ด', icon: <DashboardIcon />, path: '/' },
    { text: 'กราฟราคา', icon: <ChartIcon />, path: '/chart' },
    { text: 'พอร์ตโฟลิโอ', icon: <PortfolioIcon />, path: '/portfolio' },
    { text: 'บอทของฉัน', icon: <BotIcon />, path: '/bots' },
    { text: 'ตั้งค่า', icon: <SettingsIcon />, path: '/settings' }
  ];

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 240,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
          backgroundColor: '#1a1a1a',
          color: '#fff'
        }
      }}
    >
      <Box sx={{ p: 2 }}>
        <img src="/logo.png" alt="Logo" style={{ width: '100%' }} />
      </Box>
      <Divider sx={{ backgroundColor: '#333' }} />
      <List>
        {menuItems.map((item) => (
          <ListItem
            button
            key={item.text}
            component={Link}
            to={item.path}
            selected={location.pathname === item.path}
            sx={{
              '&.Mui-selected': {
                backgroundColor: '#333',
                '&:hover': {
                  backgroundColor: '#444'
                }
              },
              '&:hover': {
                backgroundColor: '#222'
              }
            }}
          >
            <ListItemIcon sx={{ color: '#fff' }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default Sidebar; 