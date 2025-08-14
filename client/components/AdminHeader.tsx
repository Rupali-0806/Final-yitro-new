import React from "react";
import { useAuth } from "./RealAuthProvider";
import { Button } from "./ui/button";
import { Avatar, AvatarFallback } from "./ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "./ui/dropdown-menu";
import {
  Settings,
  LogOut,
  Moon,
  Sun,
  Shield,
  User,
  Users,
  Building2,
} from "lucide-react";

interface AdminHeaderProps {
  isDarkMode?: boolean;
  onToggleDarkMode?: (enabled: boolean) => void;
  onSettingsClick?: () => void;
}

export function AdminHeader({
  isDarkMode = false,
  onToggleDarkMode = () => {},
  onSettingsClick = () => {},
}: AdminHeaderProps) {
  const { user, signOut } = useAuth();

  const handleLogout = () => {
    signOut();
    window.location.href = "/login";
  };

  return (
    <div className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
      {/* Main Header Bar */}
      <div className="px-6 py-3 flex items-center justify-between">
        {/* Left side - Logo and Title */}
        <div className="flex items-center space-x-4">
          <div className="bg-white rounded px-2 py-1">
            <img
              src="https://cdn.builder.io/api/v1/image/assets%2F087df647f1e8465b80d17ed1202a1a86%2Fe158bed7731d41aa84ba65ca872152aa?format=webp&width=800"
              alt="Yitro"
              className="h-8 w-auto"
            />
          </div>
          <div className="flex items-center space-x-2">
            <Shield className="h-5 w-5 text-blue-600" />
            <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Admin Dashboard
            </span>
          </div>
        </div>

        {/* Right side - Actions and User Menu */}
        <div className="flex items-center space-x-4">
          {/* Dark Mode Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onToggleDarkMode(!isDarkMode)}
            className="hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            {isDarkMode ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </Button>

          {/* Settings */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onSettingsClick}
            className="hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <Settings className="h-5 w-5" />
          </Button>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <div className="flex items-center space-x-3 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg p-2 transition-colors">
                <Avatar className="h-8 w-8 border border-gray-200 dark:border-gray-700">
                  <AvatarFallback className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 text-sm font-medium">
                    {user?.displayName?.charAt(0).toUpperCase() ||
                      user?.email?.charAt(0).toUpperCase() ||
                      "A"}
                  </AvatarFallback>
                </Avatar>
                <div className="hidden md:block">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {user?.displayName || user?.email?.split("@")[0] || "Admin"}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Administrator
                  </p>
                </div>
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <div className="px-3 py-2 border-b">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {user?.displayName ||
                    user?.email?.split("@")[0] ||
                    "Admin User"}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  {user?.email || "admin@yitro.com"}
                </p>
                <span className="inline-block px-2 py-1 text-xs font-semibold rounded-full mt-1 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400">
                  <Shield className="w-3 h-3 inline mr-1" />
                  Administrator
                </span>
              </div>

              <DropdownMenuItem onClick={onSettingsClick}>
                <User className="h-4 w-4 mr-2" />
                Admin Settings
              </DropdownMenuItem>

              <DropdownMenuSeparator />

              <DropdownMenuItem onClick={() => onToggleDarkMode(!isDarkMode)}>
                {isDarkMode ? (
                  <Sun className="h-4 w-4 mr-2" />
                ) : (
                  <Moon className="h-4 w-4 mr-2" />
                )}
                {isDarkMode ? "Light Mode" : "Dark Mode"}
              </DropdownMenuItem>

              <DropdownMenuSeparator />

              <DropdownMenuItem
                onClick={handleLogout}
                className="text-red-600 dark:text-red-400 focus:text-red-600 dark:focus:text-red-400"
              >
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
}
