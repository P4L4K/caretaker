import React from 'react';
import { ConfigProvider, theme } from 'antd';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import RealTimeMonitor from './components/RealTimeMonitor';

const App: React.FC = () => {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 8,
        },
        components: {
          Card: {
            headerBg: '#1f1f1f',
            colorBgContainer: '#141414',
          },
          Layout: {
            headerBg: '#1f1f1f',
            bodyBg: '#141414',
          },
        },
      }}
    >
      <div className="min-h-screen bg-gray-900 text-white">
        <Router>
          <Routes>
            <Route path="/" element={<RealTimeMonitor />} />
          </Routes>
        </Router>
      </div>
    </ConfigProvider>
  );
};

export default App;
