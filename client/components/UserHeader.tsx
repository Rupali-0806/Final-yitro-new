import React, { useState } from "react";
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
  User,
  Activity,
  BarChart3,
  FileText,
  Home,
  Target,
  Building2,
  Users,
  TrendingUp,
  MessageSquare,
  Plus,
  Search,
} from "lucide-react";

interface UserHeaderProps {
  isDarkMode?: boolean;
  onToggleDarkMode?: (enabled: boolean) => void;
  onSettingsClick?: () => void;
}

interface UserHeaderProps {
  isDarkMode?: boolean;
  onToggleDarkMode?: (enabled: boolean) => void;
  onSettingsClick?: () => void;
  currentTab?: string;
  onTabChange?: (tab: string) => void;
}

export function UserHeader({
  isDarkMode = false,
  onToggleDarkMode = () => {},
  onSettingsClick = () => {},
  currentTab = "home",
  onTabChange = () => {},
}: UserHeaderProps) {
  const { user, signOut } = useAuth();

  const handleLogout = () => {
    signOut();
    window.location.href = "/login";
  };

  // CRM Navigation Tabs
  const crmTabs = [
    { id: "home", label: "Home", icon: Home },
    { id: "leads", label: "Leads", icon: Target },
    { id: "accounts", label: "Accounts", icon: Building2 },
    { id: "contacts", label: "Contacts", icon: Users },
    { id: "active-deals", label: "Active Deals", icon: TrendingUp },
    { id: "reports", label: "Reports", icon: BarChart3 },
  ];

  const handleAddNew = (type: string) => {
    console.log("Adding new item of type:", type);
    // Switch to the appropriate tab and trigger the new item creation
    onTabChange(type);

    // Trigger new item dialog after a small delay to ensure tab is switched
    setTimeout(() => {
      console.log("Triggering new item event for:", type);
      const event = new CustomEvent("triggerNewItem", { detail: { type } });
      window.dispatchEvent(event);
    }, 100);
  };

  return (
    <div className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
      {/* Main Header Bar */}
      <div className="px-6 py-3 flex items-center justify-between">
        {/* Left side - Logo and Title */}
        <div className="flex items-center space-x-4">
          <img
            src="https://cdn.builder.io/api/v1/image/assets%2F087df647f1e8465b80d17ed1202a1a86%2Fe158bed7731d41aa84ba65ca872152aa?format=webp&width=800"
            alt="Yitro"
            className="h-8 w-auto"
          />
          <div className="flex items-center space-x-2">
            <User className="h-5 w-5 text-blue-600" />
            <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              My Dashboard
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
                  <AvatarFallback className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 text-sm font-medium">
                    {user?.displayName?.charAt(0).toUpperCase() ||
                      user?.email?.charAt(0).toUpperCase() ||
                      "U"}
                  </AvatarFallback>
                </Avatar>
                <div className="hidden md:block">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {user?.displayName || user?.email?.split("@")[0] || "User"}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Team Member
                  </p>
                </div>
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <div className="px-3 py-2 border-b">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {user?.displayName || user?.email?.split("@")[0] || "User"}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  {user?.email || "user@yitro.com"}
                </p>
                <span className="inline-block px-2 py-1 text-xs font-semibold rounded-full mt-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                  <User className="w-3 h-3 inline mr-1" />
                  Team Member
                </span>
              </div>

              <DropdownMenuItem onClick={onSettingsClick}>
                <User className="h-4 w-4 mr-2" />
                Profile
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

      {/* CRM Tab Navigation */}
      <div className="px-6">
        <nav className="flex space-x-1 overflow-x-auto">
          {crmTabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  console.log("Switching to tab:", tab.id);
                  onTabChange(tab.id);
                }}
                className={`
                  flex items-center space-x-2 px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-colors
                  ${
                    currentTab === tab.id || currentTab.startsWith(tab.id)
                      ? "border-blue-600 text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20"
                      : "border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:border-gray-300"
                  }
                `}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
