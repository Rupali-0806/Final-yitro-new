import { useState, useEffect } from "react";
import {
  BarChart3,
  TrendingUp,
  Users,
  Target,
  DollarSign,
  Calendar,
  MessageSquare,
  Clock,
} from "lucide-react";
import { useDashboardMetrics } from "@/hooks/useDashboardMetrics";
import { api } from "@/services/demoApi";

interface DashboardProps {
  onNavigate: (section: string) => void;
  userRole?: "admin" | "user" | null;
}

export default function SalesforceDashboard({
  onNavigate,
  userRole,
}: DashboardProps) {
  const dashboardMetrics = useDashboardMetrics(userRole);
  const [recentActivities, setRecentActivities] = useState<any[]>([]);
  const [upcomingTasks, setUpcomingTasks] = useState<any[]>([]);
  const [pipelineData, setPipelineData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const metrics = [
    {
      title: "Total Revenue",
      value: dashboardMetrics.totalRevenue,
      change: "+12.5%",
      trend: "up",
      icon: DollarSign,
      color: "green",
    },
    {
      title: "Open Active Deals",
      value: dashboardMetrics.openActiveDeals,
      change: "+8.2%",
      trend: "up",
      icon: Target,
      color: "blue",
    },
    {
      title: "New Leads",
      value: dashboardMetrics.newLeads,
      change: "+15.3%",
      trend: "up",
      icon: Users,
      color: "purple",
    },
    {
      title: "Closed Deals",
      value: dashboardMetrics.closedDeals,
      change: "+5.7%",
      trend: "up",
      icon: TrendingUp,
      color: "orange",
    },
  ];

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true);
        const [activitiesResponse, dealsResponse, contactsResponse] =
          await Promise.all([
            api.activities.getAll(),
            api.deals.getAll(),
            api.contacts.getAll(),
          ]);

        // Process recent activities
        if (activitiesResponse.success) {
          const activities = activitiesResponse.data
            .sort(
              (a, b) =>
                new Date(b.dateTime).getTime() - new Date(a.dateTime).getTime(),
            )
            .slice(0, 4)
            .map((activity) => ({
              type: activity.activityType?.toLowerCase() || "activity",
              description:
                activity.summary ||
                `${activity.activityType} with ${activity.associatedContact}`,
              time: getRelativeTime(activity.dateTime),
              user: activity.createdBy || "System",
            }));
          setRecentActivities(activities);
        }

        // Generate upcoming tasks from recent deals and contacts
        const tasks = [];
        if (dealsResponse.success) {
          dealsResponse.data.slice(0, 2).forEach((deal) => {
            if (
              deal.stage &&
              !["Order Won", "Order Lost"].includes(deal.stage)
            ) {
              tasks.push({
                task: `Follow up on ${deal.dealName}`,
                due: "Today, 2:00 PM",
                priority: "high",
                assignee: deal.dealOwner || "You",
              });
            }
          });
        }

        if (contactsResponse.success) {
          contactsResponse.data.slice(0, 1).forEach((contact) => {
            tasks.push({
              task: `Call ${contact.firstName} ${contact.lastName}`,
              due: "Tomorrow, 10:00 AM",
              priority: "medium",
              assignee: contact.owner || "You",
            });
          });
        }

        setUpcomingTasks(tasks.slice(0, 3));

        // Calculate pipeline data
        if (dealsResponse.success) {
          const stages = [
            "Prospecting",
            "Qualification",
            "Proposal",
            "Negotiation",
            "Closed Won",
          ];
          const pipeline = stages.map((stage) => {
            const stageDeals = dealsResponse.data.filter(
              (deal) => deal.stage === stage,
            );
            const totalValue = stageDeals.reduce((sum, deal) => {
              const value = parseFloat(
                deal.dealValue?.toString().replace(/[^0-9.-]+/g, "") || "0",
              );
              return sum + (isNaN(value) ? 0 : value);
            }, 0);

            return {
              stage,
              count: stageDeals.length,
              value: totalValue > 0 ? `$${totalValue.toLocaleString()}` : "$0",
            };
          });
          setPipelineData(pipeline);
        }
      } catch (error) {
        console.error("Error loading dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, [userRole]);

  const getRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMs = now.getTime() - date.getTime();
    const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
    const diffInDays = Math.floor(diffInHours / 24);

    if (diffInDays > 0) {
      return `${diffInDays} day${diffInDays > 1 ? "s" : ""} ago`;
    } else if (diffInHours > 0) {
      return `${diffInHours} hour${diffInHours > 1 ? "s" : ""} ago`;
    } else {
      return "Just now";
    }
  };

  const quickActions = [
    {
      label: "New Lead",
      action: () => onNavigate("leads-detail"),
      icon: Target,
      color: "blue",
    },
    {
      label: "New Account",
      action: () => onNavigate("accounts-detail"),
      icon: Users,
      color: "green",
    },
    {
      label: "New Active Deal",
      action: () => onNavigate("active-deals-detail"),
      icon: TrendingUp,
      color: "purple",
    },
    {
      label: "Log Activity",
      action: () => onNavigate("activities-detail"),
      icon: MessageSquare,
      color: "orange",
    },
  ];

  return (
    <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      {/* Welcome Section */}
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Welcome back!
          </h1>
          {userRole && (
            <span
              className={`px-3 py-1 text-xs font-semibold rounded-full ${
                userRole === "admin"
                  ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                  : "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400"
              }`}
            >
              {userRole === "admin" ? "Admin View" : "User View"}
            </span>
          )}
        </div>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          {userRole === "admin"
            ? "Here's the company-wide performance overview."
            : "Here's what's happening with your sales today."}
        </p>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                    {metric.title}
                  </p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
                    {dashboardMetrics.loading ? (
                      <span className="animate-pulse">Loading...</span>
                    ) : (
                      metric.value || "0"
                    )}
                  </p>
                </div>
                <div
                  className={`p-3 rounded-full bg-${metric.color}-100 dark:bg-${metric.color}-900`}
                >
                  <Icon
                    className={`h-6 w-6 text-${metric.color}-600 dark:text-${metric.color}-400`}
                  />
                </div>
              </div>
              <p
                className={`text-sm mt-1 ${metric.trend === "up" ? "text-green-600" : "text-red-600"}`}
              >
                {metric.change} from last month
              </p>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Actions */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Quick Actions
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {quickActions.map((action, index) => {
              const Icon = action.icon;
              return (
                <button
                  key={index}
                  onClick={action.action}
                  className={`p-4 rounded-lg border-2 border-dashed border-${action.color}-200 dark:border-${action.color}-800 hover:border-${action.color}-400 dark:hover:border-${action.color}-600 hover:bg-${action.color}-50 dark:hover:bg-${action.color}-900/20 transition-colors group`}
                >
                  <Icon
                    className={`h-6 w-6 text-${action.color}-600 dark:text-${action.color}-400 mx-auto mb-2`}
                  />
                  <p
                    className={`text-sm font-medium text-${action.color}-800 dark:text-${action.color}-300`}
                  >
                    {action.label}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Recent Activities */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Recent Activities
            </h3>
            <button
              onClick={() => onNavigate("activities")}
              className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
            >
              View all
            </button>
          </div>
          <div className="space-y-3">
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-gray-300 dark:bg-gray-600 rounded-full mt-2 flex-shrink-0 animate-pulse"></div>
                    <div className="flex-1 space-y-1">
                      <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded animate-pulse"></div>
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-3/4 animate-pulse"></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : recentActivities.length > 0 ? (
              recentActivities.map((activity, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-900 dark:text-gray-100">
                      {activity.description}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {activity.time} • {activity.user}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                No recent activities
              </p>
            )}
          </div>
        </div>

        {/* Upcoming Tasks */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Upcoming Tasks
            </h3>
            <Clock className="h-5 w-5 text-gray-400" />
          </div>
          <div className="space-y-3">
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-gray-300 dark:bg-gray-600 rounded-full mt-2 flex-shrink-0 animate-pulse"></div>
                    <div className="flex-1 space-y-1">
                      <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded animate-pulse"></div>
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3 animate-pulse"></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : upcomingTasks.length > 0 ? (
              upcomingTasks.map((task, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div
                    className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                      task.priority === "high" ? "bg-red-500" : "bg-yellow-500"
                    }`}
                  ></div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-900 dark:text-gray-100">
                      {task.task}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {task.due} • {task.assignee}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                No upcoming tasks
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Sales Pipeline Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Sales Pipeline
          </h3>
          <BarChart3 className="h-5 w-5 text-gray-400" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {loading
            ? [1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="text-center">
                  <div className="bg-gray-200 dark:bg-gray-700 rounded-lg p-4 mb-2 animate-pulse">
                    <div className="h-8 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
                  </div>
                  <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
                </div>
              ))
            : pipelineData.map((stage, index) => (
                <div key={index} className="text-center">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-2">
                    <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {stage.count}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {stage.stage}
                    </p>
                  </div>
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {stage.value}
                  </p>
                </div>
              ))}
        </div>
      </div>
    </div>
  );
}
