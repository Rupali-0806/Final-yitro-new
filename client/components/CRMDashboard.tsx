import React, { useState, useEffect } from "react";
import { useAuth } from "./RealAuthProvider";
import { UserHeader } from "./UserHeader";
import { UserSettings } from "./UserSettings";
import { CRMLeads } from "./CRMLeads";
import { CRMAccounts } from "./CRMAccounts";
import { CRMReports } from "./CRMReports";
import { CRMActiveDeals } from "./CRMActiveDeals";
import { CRMContacts } from "./CRMContacts";
import { useTheme } from "./ThemeProvider";
import { useCRM } from "../contexts/CRMContext";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  Target,
  Building2,
  Users,
  TrendingUp,
  MessageSquare,
  BarChart3,
  DollarSign,
  Calendar,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  FileText,
  Phone,
  Mail,
  Plus,
} from "lucide-react";

interface CRMDashboardProps {
  onSettingsClick?: () => void;
}

export function CRMDashboard({ onSettingsClick }: CRMDashboardProps) {
  const { user } = useAuth();
  const { isDarkMode, toggleDarkMode } = useTheme();
  const { leads, accounts, contacts, deals } = useCRM();
  const [currentTab, setCurrentTab] = useState("home");
  const [showSettings, setShowSettings] = useState(false);
  const [forceRefresh, setForceRefresh] = useState(0);

  // Auto-cleanup expired tasks every minute
  useEffect(() => {
    const cleanupInterval = setInterval(() => {
      // Force refresh to remove expired tasks
      setForceRefresh((prev) => prev + 1);
      console.log(
        "Checking for expired tasks...",
        new Date().toLocaleTimeString(),
      );
    }, 60000); // Check every minute

    return () => clearInterval(cleanupInterval);
  }, []);

  // Calculate metrics directly from CRM context data
  const crmMetrics = {
    totalRevenue: deals
      .filter((deal) => deal.stage === "Order Won")
      .reduce((sum, deal) => sum + deal.dealValue, 0),
    activeDeals: deals.filter(
      (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
    ).length,
    newLeads: leads.filter((lead) => lead.status === "New").length,
    closedDeals: deals.filter((deal) => deal.stage === "Order Won").length,
    myAccounts: accounts.length,
    myContacts: contacts.length,
    totalLeads: leads.length,
    totalAccounts: accounts.length,
    totalContacts: contacts.length,
    totalDeals: deals.length,
  };

  const handleSettingsClick = () => {
    setShowSettings(true);
  };

  const handleBackToDashboard = () => {
    setShowSettings(false);
  };

  // Generate recent activities from actual CRM data
  const recentActivities = [
    // Recent deals activity
    ...deals.slice(0, 2).map((deal, index) => ({
      id: `deal-${deal.id}`,
      type: "deal",
      title: `${deal.stage === "Order Won" ? "Deal closed" : "Deal updated"}: ${deal.dealName}`,
      account: deal.associatedAccount,
      time: deal.stage === "Order Won" ? "Recently closed" : "In progress",
      icon: deal.stage === "Order Won" ? DollarSign : TrendingUp,
    })),
    // Recent leads activity
    ...leads.slice(0, 1).map((lead) => ({
      id: `lead-${lead.id}`,
      type: "lead",
      title: `New lead: ${lead.name} from ${lead.company}`,
      contact: lead.name,
      time: lead.lastActivity,
      icon: Target,
    })),
    // Recent contacts activity
    ...contacts.slice(0, 1).map((contact) => ({
      id: `contact-${contact.id}`,
      type: "contact",
      title: `Contact updated: ${contact.firstName} ${contact.lastName}`,
      account: contact.associatedAccount,
      time: "Recently updated",
      icon: Users,
    })),
  ].slice(0, 4); // Limit to 4 most recent activities

  // Helper function to check if a task is expired
  const isTaskExpired = (dueDate: string | Date) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0); // Reset time to start of day for accurate comparison

    let taskDate: Date;
    if (typeof dueDate === "string") {
      if (dueDate === "Today") {
        taskDate = new Date();
        taskDate.setHours(0, 0, 0, 0);
      } else {
        taskDate = new Date(dueDate);
      }
    } else {
      taskDate = new Date(dueDate);
    }

    return taskDate < today;
  };

  // Helper function to format due date display
  const formatDueDate = (dueDate: Date) => {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    // Reset time for comparison
    today.setHours(0, 0, 0, 0);
    tomorrow.setHours(0, 0, 0, 0);
    const taskDate = new Date(dueDate);
    taskDate.setHours(0, 0, 0, 0);

    if (taskDate.getTime() === today.getTime()) {
      return "Today";
    } else if (taskDate.getTime() === tomorrow.getTime()) {
      return "Tomorrow";
    } else {
      return taskDate.toLocaleDateString("en-US", {
        weekday: "short",
        month: "short",
        day: "numeric",
      });
    }
  };

  // Generate upcoming tasks from actual deals and leads data
  const allGeneratedTasks = [
    // Tasks from active deals
    ...deals
      .filter((deal) => !["Order Won", "Order Lost"].includes(deal.stage))
      .map((deal) => {
        const closingDate = new Date(deal.closingDate);
        return {
          id: `task-deal-${deal.id}`,
          task: `${deal.nextStep} - ${deal.dealName}`,
          dueDate: formatDueDate(closingDate),
          actualDueDate: closingDate,
          priority:
            deal.probability > 75
              ? "High"
              : deal.probability > 50
                ? "Medium"
                : "Low",
          type: "deal",
        };
      }),
    // Tasks from new leads (due today)
    ...leads
      .filter((lead) => lead.status === "New")
      .map((lead) => {
        const today = new Date();
        return {
          id: `task-lead-${lead.id}`,
          task: `Follow up with ${lead.name} at ${lead.company}`,
          dueDate: "Today",
          actualDueDate: today,
          priority: lead.score > 80 ? "High" : "Medium",
          type: "call",
        };
      }),
  ];

  // Filter out expired tasks and limit to 3 most recent
  const upcomingTasks = allGeneratedTasks
    .filter((task) => !isTaskExpired(task.actualDueDate))
    .sort(
      (a, b) =>
        new Date(a.actualDueDate).getTime() -
        new Date(b.actualDueDate).getTime(),
    )
    .slice(0, 3);

  const [showTaskDialog, setShowTaskDialog] = useState(false);
  const [newTask, setNewTask] = useState({
    title: "",
    dueDate: "",
    priority: "Medium",
    type: "task",
    description: "",
  });

  const handleAddTask = () => {
    setShowTaskDialog(true);
    console.log("Add new task dialog opened");
  };

  const handleSaveTask = () => {
    console.log("Saving new task:", newTask);
    // Here you would typically save the task to your state or backend
    setShowTaskDialog(false);
    setNewTask({
      title: "",
      dueDate: "",
      priority: "Medium",
      type: "task",
      description: "",
    });
    alert(`Task "${newTask.title}" has been created successfully!`);
  };

  const handleCancelTask = () => {
    setShowTaskDialog(false);
    setNewTask({
      title: "",
      dueDate: "",
      priority: "Medium",
      type: "task",
      description: "",
    });
  };

  // Calculate dynamic sales pipeline from actual deals data using correct stage names
  const salesPipeline = [
    {
      stage: "Opportunity Identified",
      count: deals.filter((deal) => deal.stage === "Opportunity Identified")
        .length,
      value: deals
        .filter((deal) => deal.stage === "Opportunity Identified")
        .reduce((sum, deal) => sum + deal.dealValue, 0),
    },
    {
      stage: "Proposal Submitted",
      count: deals.filter((deal) => deal.stage === "Proposal Submitted").length,
      value: deals
        .filter((deal) => deal.stage === "Proposal Submitted")
        .reduce((sum, deal) => sum + deal.dealValue, 0),
    },
    {
      stage: "Negotiating",
      count: deals.filter((deal) => deal.stage === "Negotiating").length,
      value: deals
        .filter((deal) => deal.stage === "Negotiating")
        .reduce((sum, deal) => sum + deal.dealValue, 0),
    },
    {
      stage: "Closing",
      count: deals.filter((deal) => deal.stage === "Closing").length,
      value: deals
        .filter((deal) => deal.stage === "Closing")
        .reduce((sum, deal) => sum + deal.dealValue, 0),
    },
    {
      stage: "Order Won",
      count: deals.filter((deal) => deal.stage === "Order Won").length,
      value: deals
        .filter((deal) => deal.stage === "Order Won")
        .reduce((sum, deal) => sum + deal.dealValue, 0),
    },
  ];

  const renderHome = () => (
    <div className="p-6 space-y-6">
      {/* Welcome Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Welcome back, {user?.displayName || "User"}!
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Here's what's happening with your sales today.
          </p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${crmMetrics.totalRevenue.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground flex items-center">
              <ArrowUpRight className="h-3 w-3 mr-1 text-green-500" />
              +12% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Deals</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{crmMetrics.activeDeals}</div>
            <p className="text-xs text-muted-foreground">
              Pipeline value: $
              {deals
                .filter(
                  (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
                )
                .reduce((sum, deal) => sum + deal.dealValue, 0)
                .toLocaleString()}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">New Leads</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{crmMetrics.newLeads}</div>
            <p className="text-xs text-muted-foreground">
              {leads.length} total leads
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Closed Deals</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{crmMetrics.closedDeals}</div>
            <p className="text-xs text-muted-foreground">This month</p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button
              variant="outline"
              className="h-20 flex flex-col items-center justify-center space-y-2"
              onClick={() => setCurrentTab("leads")}
            >
              <Target className="h-6 w-6" />
              <span className="text-sm">New Lead</span>
            </Button>
            <Button
              variant="outline"
              className="h-20 flex flex-col items-center justify-center space-y-2"
              onClick={() => setCurrentTab("accounts")}
            >
              <Building2 className="h-6 w-6" />
              <span className="text-sm">New Account</span>
            </Button>
            <Button
              variant="outline"
              className="h-20 flex flex-col items-center justify-center space-y-2"
              onClick={() => setCurrentTab("active-deals")}
            >
              <TrendingUp className="h-6 w-6" />
              <span className="text-sm">New Active Deal</span>
            </Button>
            <Button
              variant="outline"
              className="h-20 flex flex-col items-center justify-center space-y-2"
              onClick={() => setCurrentTab("reports")}
            >
              <BarChart3 className="h-6 w-6" />
              <span className="text-sm">View Reports</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activities */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Recent Activities
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivities.map((activity) => {
                const Icon = activity.icon;
                return (
                  <div key={activity.id} className="flex items-start space-x-3">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                        <Icon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {activity.title}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {activity.account ||
                          activity.contact ||
                          activity.opportunity ||
                          activity.amount}
                      </p>
                      <p className="text-xs text-gray-400 dark:text-gray-500">
                        {activity.time}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Upcoming Tasks */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Upcoming Tasks
              <Button variant="ghost" size="sm" onClick={handleAddTask}>
                <Plus className="h-4 w-4" />
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {upcomingTasks.length > 0 ? (
                upcomingTasks.map((task) => (
                  <div
                    key={task.id}
                    className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                  >
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {task.task}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center">
                        <Clock className="h-3 w-3 mr-1" />
                        {task.dueDate}
                      </p>
                    </div>
                    <Badge
                      variant={
                        task.priority === "High" ? "destructive" : "secondary"
                      }
                      className="text-xs"
                    >
                      {task.priority}
                    </Badge>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <Clock className="h-8 w-8 mx-auto mb-2 text-gray-400 dark:text-gray-600" />
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    No upcoming tasks
                  </p>
                  <p className="text-xs text-gray-400 dark:text-gray-500">
                    All tasks are completed or expired
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Sales Pipeline */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Sales Pipeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {salesPipeline.map((stage, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {stage.stage}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {stage.count} active deals
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      ${stage.value.toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  const renderTabContent = () => {
    console.log("Rendering tab content for:", currentTab);
    switch (currentTab) {
      case "home":
        return renderHome();
      case "leads":
        return <CRMLeads />;
      case "accounts":
        return <CRMAccounts />;
      case "contacts":
        return <CRMContacts />;
      case "active-deals":
        return <CRMActiveDeals />;
      case "reports":
        return <CRMReports />;
      default:
        console.log("Unknown tab, rendering home:", currentTab);
        return renderHome();
    }
  };

  if (showSettings) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <UserHeader
          isDarkMode={isDarkMode}
          onToggleDarkMode={toggleDarkMode}
          onSettingsClick={handleSettingsClick}
          currentTab={currentTab}
          onTabChange={setCurrentTab}
        />
        <UserSettings onBack={handleBackToDashboard} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <UserHeader
        isDarkMode={isDarkMode}
        onToggleDarkMode={toggleDarkMode}
        onSettingsClick={handleSettingsClick}
        currentTab={currentTab}
        onTabChange={setCurrentTab}
      />
      {renderTabContent()}

      {/* Task Creation Dialog */}
      <Dialog open={showTaskDialog} onOpenChange={setShowTaskDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Add New Task</DialogTitle>
            <DialogDescription>
              Create a new task to track your upcoming activities and deadlines.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="task-title" className="text-right">
                Title
              </Label>
              <Input
                id="task-title"
                placeholder="Enter task title"
                className="col-span-3"
                value={newTask.title}
                onChange={(e) =>
                  setNewTask({ ...newTask, title: e.target.value })
                }
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="task-due-date" className="text-right">
                Due Date
              </Label>
              <Input
                id="task-due-date"
                type="date"
                className="col-span-3"
                value={newTask.dueDate}
                onChange={(e) =>
                  setNewTask({ ...newTask, dueDate: e.target.value })
                }
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="task-priority" className="text-right">
                Priority
              </Label>
              <Select
                value={newTask.priority}
                onValueChange={(value) =>
                  setNewTask({ ...newTask, priority: value })
                }
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select priority" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Low">Low</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="task-type" className="text-right">
                Type
              </Label>
              <Select
                value={newTask.type}
                onValueChange={(value) =>
                  setNewTask({ ...newTask, type: value })
                }
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select task type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="call">Call</SelectItem>
                  <SelectItem value="email">Email</SelectItem>
                  <SelectItem value="meeting">Meeting</SelectItem>
                  <SelectItem value="task">Task</SelectItem>
                  <SelectItem value="follow-up">Follow-up</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="task-description" className="text-right">
                Description
              </Label>
              <Input
                id="task-description"
                placeholder="Optional description"
                className="col-span-3"
                value={newTask.description}
                onChange={(e) =>
                  setNewTask({ ...newTask, description: e.target.value })
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={handleCancelTask}>
              Cancel
            </Button>
            <Button
              type="submit"
              onClick={handleSaveTask}
              disabled={!newTask.title}
            >
              Save Task
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
