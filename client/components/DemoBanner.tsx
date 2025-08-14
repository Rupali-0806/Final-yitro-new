import { useState } from 'react';
import { X, Globe, Database } from 'lucide-react';

export default function DemoBanner() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  const isDemoMode = window.location.hostname.includes('netlify') || 
                     window.location.hostname.includes('vercel') ||
                     !window.location.hostname.includes('localhost');

  if (!isDemoMode) return null;

  return (
    <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-3 relative">
      <div className="max-w-6xl mx-auto flex items-center justify-center space-x-4">
        <Globe className="h-5 w-5" />
        <div className="text-center">
          <p className="text-sm font-medium">
            🎭 <strong>Live Demo:</strong> Yitro CRM Platform - Professional Sales Management System
          </p>
          <p className="text-xs opacity-90 mt-1">
            ✨ All features functional with demo data • Real-time UI • Complete CRM capabilities
          </p>
        </div>
        <Database className="h-5 w-5" />
        
        <button
          onClick={() => setIsVisible(false)}
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/80 hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
