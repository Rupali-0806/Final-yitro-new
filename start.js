#!/usr/bin/env node

/**
 * Yitro CRM Platform - Production Startup Script
 * 
 * This script starts the production CRM server with all necessary configurations.
 * It handles database connections, server setup, and error handling.
 */

const express = require('express');
const path = require('path');
const { createServer } = require('./dist/server/node-build.mjs');

const PORT = process.env.PORT || 8080;
const NODE_ENV = process.env.NODE_ENV || 'production';

console.log('🚀 Starting Yitro CRM Platform...');
console.log(`📊 Environment: ${NODE_ENV}`);
console.log(`🌐 Port: ${PORT}`);

try {
  // Create the server
  const app = createServer();
  
  // Serve static files in production
  if (NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, 'dist/spa')));
    
    // Handle client-side routing
    app.get('*', (req, res) => {
      // Skip API routes
      if (req.path.startsWith('/api/')) {
        return res.status(404).json({ error: 'API endpoint not found' });
      }
      
      res.sendFile(path.join(__dirname, 'dist/spa/index.html'));
    });
  }
  
  // Start the server
  app.listen(PORT, () => {
    console.log('✅ Yitro CRM Platform is running!');
    console.log(`🌍 Access your CRM at: http://localhost:${PORT}`);
    console.log('📱 Features available:');
    console.log('   • Complete CRM functionality');
    console.log('   • User profile management');
    console.log('   • Real-time dashboard metrics');
    console.log('   • Professional reports');
    console.log('   • Dark/Light mode themes');
    console.log('   • Responsive mobile design');
    console.log('');
    console.log('🎯 Ready for production use!');
  });
  
} catch (error) {
  console.error('❌ Failed to start Yitro CRM Platform:');
  console.error(error.message);
  console.error('');
  console.error('📋 Troubleshooting steps:');
  console.error('1. Run "npm install" to ensure dependencies are installed');
  console.error('2. Run "npm run build" to build the application');
  console.error('3. Check that DATABASE_URL is configured correctly');
  console.error('4. Verify that port', PORT, 'is available');
  process.exit(1);
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Shutting down Yitro CRM Platform gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('🛑 Shutting down Yitro CRM Platform gracefully...');
  process.exit(0);
});
