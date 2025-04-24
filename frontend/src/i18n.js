import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  en: {
    translation: {
      'dashboard': 'Dashboard',
      'chart': 'Price Chart',
      'portfolio': 'Portfolio',
      'my_bots': 'My Bots',
      'settings': 'Settings',
      'create_bot': 'Create Bot',
      'bot_id': 'Bot ID',
      'api_key': 'API Key',
      'api_secret': 'API Secret',
      'symbols': 'Symbols',
      'profit_target': 'Profit Target (%)',
      'buy_dip': 'Buy Dip (%)',
      'max_position': 'Max Position (%)',
      'max_trade': 'Max Trade (%)',
      'status': 'Status',
      'running': 'Running',
      'stopped': 'Stopped',
      'start': 'Start',
      'stop': 'Stop',
      'delete': 'Delete',
      'save': 'Save',
      'cancel': 'Cancel',
      'error': 'Error',
      'success': 'Success',
      'my_account': 'My Account',
      'logout': 'Logout',
      'strategies': 'Strategies',
      'bot_control': 'Bot Control'
    }
  },
  th: {
    translation: {
      'dashboard': 'แดชบอร์ด',
      'chart': 'กราฟราคา',
      'portfolio': 'พอร์ตโฟลิโอ',
      'my_bots': 'บอทของฉัน',
      'settings': 'ตั้งค่า',
      'create_bot': 'สร้างบอทใหม่',
      'bot_id': 'รหัสบอท',
      'api_key': 'API Key',
      'api_secret': 'API Secret',
      'symbols': 'สัญลักษณ์',
      'profit_target': 'เป้าหมายกำไร (%)',
      'buy_dip': 'Buy Dip (%)',
      'max_position': 'Max Position (%)',
      'max_trade': 'Max Trade (%)',
      'status': 'สถานะ',
      'running': 'กำลังทำงาน',
      'stopped': 'หยุดทำงาน',
      'start': 'เริ่ม',
      'stop': 'หยุด',
      'delete': 'ลบ',
      'save': 'บันทึก',
      'cancel': 'ยกเลิก',
      'error': 'เกิดข้อผิดพลาด',
      'success': 'สำเร็จ',
      'my_account': 'บัญชีของฉัน',
      'logout': 'ออกจากระบบ',
      'strategies': 'กลยุทธ์',
      'bot_control': 'ควบคุมบอท'
    }
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: 'en', // เปลี่ยนเป็นภาษาอังกฤษ
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n; 