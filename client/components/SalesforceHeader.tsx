import { useState } from "react";
import { useAuth } from "@/components/RealAuthProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Search,
  Plus,
  Bell,
  Settings,
  Grid3X3,
  Users,
  User,
  Activity,
  FileText,
  TrendingUp,
  MessageSquare,
  BarChart3,
  Target,
  LogOut,
} from "lucide-react";

interface SalesforceHeaderProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  onLogout: () => void;
  isDarkMode: boolean;
  onToggleDarkMode: (enabled: boolean) => void;
  onSettingsClick?: () => void;
  userRole?: "admin" | "user" | null;
}

export default function SalesforceHeader({
  currentTab,
  onTabChange,
  onLogout,
  isDarkMode,
  onToggleDarkMode,
  onSettingsClick,
  userRole,
}: SalesforceHeaderProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showNotifications, setShowNotifications] = useState(false);
  const { user } = useAuth();

  const notifications = [
    {
      id: 1,
      type: "deal",
      message: "New deal created: Tech Solutions Partnership",
      time: "5 minutes ago",
      unread: true,
    },
    {
      id: 2,
      type: "activity",
      message: "Follow-up call scheduled for tomorrow",
      time: "1 hour ago",
      unread: true,
    },
    {
      id: 3,
      type: "lead",
      message: "New lead assigned: John Smith from Acme Corp",
      time: "2 hours ago",
      unread: false,
    },
    {
      id: 4,
      type: "task",
      message: "Reminder: Proposal due for Global Industries",
      time: "3 hours ago",
      unread: false,
    },
  ];

  const unreadCount = notifications.filter((n) => n.unread).length;

  const tabs = [
    { id: "home", label: "Home", icon: Grid3X3 },
    { id: "leads", label: "Leads", icon: Target },
    { id: "accounts", label: "Accounts", icon: User },
    { id: "contacts", label: "Contacts", icon: Users },
    { id: "active-deals", label: "Active Deals", icon: TrendingUp },
    { id: "reports", label: "Reports", icon: BarChart3 },
  ];

  const handleAddNew = (type: string) => {
    onTabChange(`${type}-detail`);
  };

  return (
    <div className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
      {/* Top Bar */}
      <div className="px-4 py-2 flex items-center justify-between bg-blue-400 text-white">
        <div className="flex items-center space-x-4">
          <img
            src="https://cdn.builder.io/api/v1/image/assets%2F087df647f1e8465b80d17ed1202a1a86%2Fe158bed7731d41aa84ba65ca872152aa?format=webp&width=800"
            alt="Yitro CRM"
            className="h-8 w-auto"
          />
          <span className="text-sm font-medium">CRM Platform</span>
        </div>

        <div className="flex items-center space-x-3">
          <DropdownMenu
            open={showNotifications}
            onOpenChange={setShowNotifications}
          >
            <DropdownMenuTrigger asChild>
              <div className="relative cursor-pointer">
                <Bell className="h-5 w-5 hover:text-blue-200" />
                {unreadCount > 0 && (
                  <div className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                    {unreadCount}
                  </div>
                )}
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80">
              <div className="p-3 border-b">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  Notifications
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {unreadCount} unread
                </p>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {notifications.map((notification) => (
                  <DropdownMenuItem
                    key={notification.id}
                    className={`p-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 ${
                      notification.unread
                        ? "bg-blue-50 dark:bg-blue-900/20"
                        : ""
                    }`}
                  >
                    <div className="flex-1">
                      <p
                        className={`text-sm ${notification.unread ? "font-medium" : ""} text-gray-900 dark:text-gray-100`}
                      >
                        {notification.message}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {notification.time}
                      </p>
                    </div>
                    {notification.unread && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full ml-2"></div>
                    )}
                  </DropdownMenuItem>
                ))}
              </div>
              <div className="p-3 border-t">
                <button className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300">
                  Mark all as read
                </button>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
          <Settings
            className="h-5 w-5 cursor-pointer hover:text-blue-200"
            onClick={() => onSettingsClick && onSettingsClick()}
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Avatar className="h-8 w-8 cursor-pointer border-2 border-white/20">
                <AvatarFallback className="bg-blue-500 text-white text-sm">
                  {user?.displayName?.charAt(0).toUpperCase() ||
                    user?.email?.charAt(0).toUpperCase() ||
                    "U"}
                </AvatarFallback>
              </Avatar>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <div className="px-3 py-2 border-b">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {user?.displayName || user?.email?.split("@")[0] || "User"}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  {user?.email}
                </p>
                {userRole && (
                  <span
                    className={`inline-block px-2 py-1 text-xs font-semibold rounded-full mt-1 ${
                      userRole === "admin"
                        ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                        : "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400"
                    }`}
                  >
                    {userRole === "admin" ? "Admin" : "User"}
                  </span>
                )}
              </div>
              <DropdownMenuItem
                onClick={() => onSettingsClick && onSettingsClick()}
              >
                <Settings className="h-4 w-4 mr-2" />
                Profile & Settings
              </DropdownMenuItem>
              {userRole === "admin" && (
                <DropdownMenuItem
                  onClick={() => onTabChange("user-management")}
                >
                  <Users className="h-4 w-4 mr-2" />
                  Manage Users
                </DropdownMenuItem>
              )}
              <DropdownMenuItem onClick={() => onToggleDarkMode(!isDarkMode)}>
                <Settings className="h-4 w-4 mr-2" />
                {isDarkMode ? "Light Mode" : "Dark Mode"}
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onLogout}>
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Search and Actions Bar */}
      <div className="px-4 py-3 flex items-center justify-between">
        <div className="flex-1 max-w-2xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-600"
            />
          </div>
        </div>

        <div className="flex items-center space-x-2 ml-4">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button className="bg-blue-600 hover:bg-blue-700 text-white">
                <Plus className="h-4 w-4 mr-2" />
                New
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={() => handleAddNew("leads")}>
                <Target className="h-4 w-4 mr-2" />
                New Lead
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleAddNew("accounts")}>
                <User className="h-4 w-4 mr-2" />
                New Account
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleAddNew("contacts")}>
                <Users className="h-4 w-4 mr-2" />
                New Contact
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleAddNew("active-deals")}>
                <TrendingUp className="h-4 w-4 mr-2" />
                New Active Deal
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="px-4">
        <nav className="flex space-x-1 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
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
