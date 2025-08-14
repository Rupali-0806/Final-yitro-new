// Demo API service that falls back to mock data for frontend demos
import { api as realApi } from './api';
import { mockApi } from './mockApi';

// Check if we're in demo mode (no backend available)
const isDemoMode = () => {
  return window.location.hostname.includes('netlify') || 
         window.location.hostname.includes('vercel') ||
         !window.location.hostname.includes('localhost');
};

// Create demo API that tries real API first, falls back to mock
const createDemoAPI = () => {
  const demoApi = {} as typeof realApi;

  // For each API endpoint, create a version that tries real API first
  Object.keys(realApi).forEach(key => {
    const apiKey = key as keyof typeof realApi;
    demoApi[apiKey] = {} as any;

    if (realApi[apiKey] && typeof realApi[apiKey] === 'object') {
      Object.keys(realApi[apiKey]).forEach(method => {
        const methodKey = method as keyof typeof realApi[typeof apiKey];
        demoApi[apiKey][methodKey] = async (...args: any[]) => {
          // If in demo mode, use mock API directly
          if (isDemoMode()) {
            console.log(`🎭 Demo mode: Using mock data for ${String(apiKey)}.${String(methodKey)}`);
            return (mockApi as any)[apiKey][methodKey](...args);
          }

          // Try real API first
          try {
            console.log(`🌐 Trying real API: ${String(apiKey)}.${String(methodKey)}`);
            return await (realApi[apiKey] as any)[methodKey](...args);
          } catch (error) {
            console.warn(`⚠️ Real API failed, falling back to mock for ${String(apiKey)}.${String(methodKey)}`);
            console.warn('Error:', error);
            return (mockApi as any)[apiKey][methodKey](...args);
          }
        };
      });
    }
  });

  return demoApi;
};

export const api = createDemoAPI();
export default api;
