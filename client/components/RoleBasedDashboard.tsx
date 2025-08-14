import React, { useState } from "react";
import { useAuth } from "./RealAuthProvider";
import { AdminUserManagement } from "./AdminUserManagement";
import { AdminMetrics } from "./AdminMetrics";
import { AdminHeader } from "./AdminHeader";
import { AdminSettings } from "./AdminSettings";
import { CRMDashboard } from "./CRMDashboard";
import { useTheme } from "./ThemeProvider";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import {
  Shield,
  User,
  Users,
  Activity,
  BarChart3,
  Settings,
} from "lucide-react";

// User Dashboard - Full CRM functionality for regular users
function UserDashboard() {
  return <CRMDashboard />;
}

// Admin Dashboard - Full access for administrators
function AdminDashboard() {
  const { user } = useAuth();
  const { isDarkMode, toggleDarkMode } = useTheme();
  const [showSettings, setShowSettings] = useState(false);

  const handleSettingsClick = () => {
    setShowSettings(true);
  };

  const handleBackToDashboard = () => {
    setShowSettings(false);
  };

  if (showSettings) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <AdminHeader
          isDarkMode={isDarkMode}
          onToggleDarkMode={toggleDarkMode}
          onSettingsClick={handleSettingsClick}
        />
        <AdminSettings onBack={handleBackToDashboard} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <AdminHeader
        isDarkMode={isDarkMode}
        onToggleDarkMode={toggleDarkMode}
        onSettingsClick={handleSettingsClick}
      />

      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              Welcome back, {user?.displayName || "Admin"}
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Company-wide metrics and user management dashboard
            </p>
          </div>
        </div>

        {/* Company-wide Metrics */}
        <AdminMetrics />

        {/* Admin User Management */}
        <AdminUserManagement />
      </div>
    </div>
  );
}

// Main Role-Based Dashboard Component
export function RoleBasedDashboard() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-gray-500">
            Please log in to access the dashboard.
          </p>
        </CardContent>
      </Card>
    );
  }

  // Render dashboard based on user role
  return user.role === "admin" ? <AdminDashboard /> : <UserDashboard />;
}
