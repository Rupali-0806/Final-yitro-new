import React, { useState, useEffect } from "react";
import { useAuth } from "./RealAuthProvider";
import { useCRM } from "../contexts/CRMContext";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import {
  FileText,
  Download,
  Filter,
  Calendar,
  Users,
  Building2,
  TrendingUp,
  DollarSign,
  BarChart3,
  RefreshCw,
} from "lucide-react";

interface ReportData {
  id: string;
  name: string;
  type: "sales" | "activities" | "performance" | "leads";
  generatedAt: string;
  period: string;
  status: "completed" | "processing" | "failed";
  data: any;
}

interface UserMetrics {
  userId: string;
  userName: string;
  totalDeals: number;
  wonDeals: number;
  dealValue: number;
  activities: number;
  conversionRate: number;
  leadsGenerated: number;
}

export function CRMReports() {
  const { user } = useAuth();
  const { leads, accounts, contacts, deals } = useCRM();
  const [reports, setReports] = useState<ReportData[]>([]);
  const [userMetrics, setUserMetrics] = useState<UserMetrics[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPeriod, setSelectedPeriod] = useState("monthly");
  const [selectedType, setSelectedType] = useState("all");

  useEffect(() => {
    generateAutomaticReports();
    if (user?.role === "admin") {
      fetchAllUserMetrics();
    } else {
      fetchUserMetrics();
    }
  }, [user, selectedPeriod, deals, contacts, accounts, leads]);

  const generateAutomaticReports = async () => {
    setLoading(true);
    try {
      // Simulate API call for report generation
      await new Promise((resolve) => setTimeout(resolve, 1000));

      const mockReports: ReportData[] = [
        {
          id: "rpt-001",
          name: "Sales Performance Report",
          type: "sales",
          generatedAt: new Date().toISOString(),
          period: selectedPeriod,
          status: "completed",
          data: generateSalesData(),
        },
        {
          id: "rpt-002",
          name: "Activities Report",
          type: "activities",
          generatedAt: new Date().toISOString(),
          period: selectedPeriod,
          status: "completed",
          data: generateActivitiesData(),
        },
        {
          id: "rpt-003",
          name: "Performance Metrics",
          type: "performance",
          generatedAt: new Date().toISOString(),
          period: selectedPeriod,
          status: "completed",
          data: generatePerformanceData(),
        },
        {
          id: "rpt-004",
          name: "Leads Report",
          type: "leads",
          generatedAt: new Date().toISOString(),
          period: selectedPeriod,
          status: "completed",
          data: generateLeadsData(),
        },
      ];

      setReports(mockReports);
    } catch (error) {
      console.error("Failed to generate reports:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchUserMetrics = async () => {
    // For regular users, show only their own metrics
    const currentUserName = user?.displayName || "Current User";

    // Calculate real metrics from CRM data for current user
    const userDeals = deals.filter((deal) => deal.owner === currentUserName);
    const userWonDeals = userDeals.filter((deal) => deal.stage === "Order Won");
    const userContacts = contacts.filter(
      (contact) => contact.owner === currentUserName,
    );
    const userAccounts = accounts.filter(
      (account) => account.owner === currentUserName,
    );

    // Calculate leads generated (approximate from leads data)
    const userLeads = leads.filter(
      (lead) =>
        userAccounts.some((account) => account.name === lead.company) ||
        userContacts.some(
          (contact) => `${contact.firstName} ${contact.lastName}` === lead.name,
        ),
    );

    const totalDealValue = userWonDeals.reduce(
      (sum, deal) => sum + deal.dealValue,
      0,
    );
    const conversionRate =
      userDeals.length > 0 ? (userWonDeals.length / userDeals.length) * 100 : 0;
    const totalActivities =
      userDeals.length + userContacts.length + userAccounts.length;

    const userMetric: UserMetrics = {
      userId: user?.id || "1",
      userName: currentUserName,
      totalDeals: userDeals.length,
      wonDeals: userWonDeals.length,
      dealValue: totalDealValue,
      activities: totalActivities,
      conversionRate: parseFloat(conversionRate.toFixed(1)),
      leadsGenerated: userLeads.length,
    };

    setUserMetrics([userMetric]);
  };

  const fetchAllUserMetrics = async () => {
    // For admins, show all users' metrics calculated from real CRM data

    // Get all unique users from deals, contacts, and accounts
    const allUserNames = new Set([
      ...deals.map((deal) => deal.owner),
      ...contacts.map((contact) => contact.owner || "Unknown"),
      ...accounts.map((account) => account.owner),
    ]);

    const allUserMetrics: UserMetrics[] = Array.from(allUserNames)
      .map((userName, index) => {
        // Calculate metrics for each user
        const userDeals = deals.filter((deal) => deal.owner === userName);
        const userWonDeals = userDeals.filter(
          (deal) => deal.stage === "Order Won",
        );
        const userContacts = contacts.filter(
          (contact) => contact.owner === userName,
        );
        const userAccounts = accounts.filter(
          (account) => account.owner === userName,
        );

        // Calculate leads generated (approximate from related data)
        const userLeads = leads.filter(
          (lead) =>
            userAccounts.some((account) => account.name === lead.company) ||
            userContacts.some(
              (contact) =>
                `${contact.firstName} ${contact.lastName}` === lead.name,
            ),
        );

        const totalDealValue = userWonDeals.reduce(
          (sum, deal) => sum + deal.dealValue,
          0,
        );
        const conversionRate =
          userDeals.length > 0
            ? (userWonDeals.length / userDeals.length) * 100
            : 0;
        const totalActivities =
          userDeals.length + userContacts.length + userAccounts.length;

        return {
          userId: `user-${index + 1}`,
          userName: userName || "Unknown User",
          totalDeals: userDeals.length,
          wonDeals: userWonDeals.length,
          dealValue: totalDealValue,
          activities: totalActivities,
          conversionRate: parseFloat(conversionRate.toFixed(1)),
          leadsGenerated: userLeads.length,
        };
      })
      .filter((metric) => metric.userName !== "Unknown User"); // Filter out unknown users

    setUserMetrics(allUserMetrics);
  };

  const generateSalesData = () => {
    const wonDeals = deals.filter((deal) => deal.stage === "Order Won");
    const totalRevenue = wonDeals.reduce(
      (sum, deal) => sum + deal.dealValue,
      0,
    );
    const averageDealSize =
      wonDeals.length > 0 ? totalRevenue / wonDeals.length : 0;

    return {
      totalRevenue,
      deals: wonDeals.length,
      averageDealSize: Math.round(averageDealSize),
      monthlyGrowth: 12.5, // This would need historical data to calculate properly
    };
  };

  const generateActivitiesData = () => {
    const totalActivities =
      deals.length + contacts.length + accounts.length + leads.length;

    return {
      totalActivities,
      calls: Math.round(totalActivities * 0.35), // Approximate distribution
      meetings: Math.round(totalActivities * 0.25),
      emails: Math.round(totalActivities * 0.4),
      avgPerDay: parseFloat((totalActivities / 30).toFixed(1)), // Assuming 30-day period
    };
  };

  const generatePerformanceData = () => {
    const wonDeals = deals.filter((deal) => deal.stage === "Order Won");
    const conversionRate =
      deals.length > 0 ? (wonDeals.length / deals.length) * 100 : 0;

    return {
      conversionRate: parseFloat(conversionRate.toFixed(1)),
      avgSalesCycle: 45, // This would need historical data
      teamProductivity: Math.min(95, 60 + conversionRate / 2), // Calculated based on performance
      customerSatisfaction: 4.7, // This would come from customer feedback data
    };
  };

  const generateLeadsData = () => {
    const qualifiedLeads = leads.filter(
      (lead) => lead.status === "Qualified" || lead.status === "Working",
    );
    const hotLeads = leads.filter((lead) => lead.score >= 80);
    const conversionRate =
      leads.length > 0 ? (qualifiedLeads.length / leads.length) * 100 : 0;

    return {
      totalLeads: leads.length,
      qualifiedLeads: qualifiedLeads.length,
      hotLeads: hotLeads.length,
      conversionRate: parseFloat(conversionRate.toFixed(1)),
    };
  };

  const exportToExcel = (report: ReportData) => {
    // Create CSV content
    let csvContent = "";
    const reportTitle = `${report.name} - ${report.period}\n`;
    const generatedDate = `Generated: ${new Date(report.generatedAt).toLocaleString()}\n\n`;

    csvContent += reportTitle + generatedDate;

    if (report.type === "sales") {
      csvContent += "Sales Performance Data\n";
      csvContent += "Metric,Value\n";
      csvContent += `Total Revenue,$${report.data.totalRevenue.toLocaleString()}\n`;
      csvContent += `Total Deals,${report.data.deals}\n`;
      csvContent += `Average Deal Size,$${report.data.averageDealSize.toLocaleString()}\n`;
      csvContent += `Monthly Growth,${report.data.monthlyGrowth}%\n\n`;
    } else if (report.type === "activities") {
      csvContent += "Activities Data\n";
      csvContent += "Activity Type,Count\n";
      csvContent += `Total Activities,${report.data.totalActivities}\n`;
      csvContent += `Calls,${report.data.calls}\n`;
      csvContent += `Meetings,${report.data.meetings}\n`;
      csvContent += `Emails,${report.data.emails}\n`;
      csvContent += `Average Per Day,${report.data.avgPerDay}\n\n`;
    }

    // Add user metrics if admin
    if (user?.role === "admin" && userMetrics.length > 0) {
      csvContent += "User Performance Metrics\n";
      csvContent +=
        "User Name,Total Deals,Won Deals,Deal Value,Activities,Conversion Rate,Leads Generated\n";
      userMetrics.forEach((metric) => {
        csvContent += `${metric.userName},${metric.totalDeals},${metric.wonDeals},$${metric.dealValue.toLocaleString()},${metric.activities},${metric.conversionRate}%,${metric.leadsGenerated}\n`;
      });
    }

    // Create and download file
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `${report.name.replace(/\s+/g, "_")}_${new Date().toISOString().split("T")[0]}.csv`,
    );
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getReportIcon = (type: string) => {
    switch (type) {
      case "sales":
        return DollarSign;
      case "activities":
        return BarChart3;
      case "performance":
        return TrendingUp;
      case "leads":
        return Users;
      default:
        return FileText;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
      case "processing":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400";
      case "failed":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400";
    }
  };

  const filteredReports = reports.filter(
    (report) => selectedType === "all" || report.type === selectedType,
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <FileText className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              Reports & Analytics
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {user?.role === "admin"
                ? "Company-wide reports and user performance metrics"
                : "Your personal performance reports and analytics"}
            </p>
          </div>
        </div>
        <Button
          onClick={generateAutomaticReports}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700"
        >
          {loading ? (
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <RefreshCw className="h-4 w-4 mr-2" />
          )}
          Refresh Reports
        </Button>
      </div>

      {/* Filters */}
      <div className="flex space-x-4">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">
              <Calendar className="h-4 w-4 mr-2" />
              Period: {selectedPeriod}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setSelectedPeriod("daily")}>
              Daily
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedPeriod("weekly")}>
              Weekly
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedPeriod("monthly")}>
              Monthly
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedPeriod("quarterly")}>
              Quarterly
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">
              <Filter className="h-4 w-4 mr-2" />
              Type: {selectedType === "all" ? "All" : selectedType}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setSelectedType("all")}>
              All Reports
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedType("sales")}>
              Sales
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedType("activities")}>
              Activities
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedType("performance")}>
              Performance
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setSelectedType("leads")}>
              Leads
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Reports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filteredReports.map((report) => {
          const Icon = getReportIcon(report.type);
          return (
            <Card key={report.id}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <div className="flex items-center space-x-2">
                  <Icon className="h-5 w-5 text-blue-600" />
                  <CardTitle className="text-lg">{report.name}</CardTitle>
                </div>
                <Badge className={getStatusColor(report.status)}>
                  {report.status}
                </Badge>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">
                      Period:
                    </span>
                    <span>{report.period}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">
                      Generated:
                    </span>
                    <span>{new Date(report.generatedAt).toLocaleString()}</span>
                  </div>
                  <div className="pt-3">
                    <Button
                      onClick={() => exportToExcel(report)}
                      className="w-full"
                      disabled={report.status !== "completed"}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Export to Excel
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* User Performance Metrics */}
      {userMetrics.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>
                {user?.role === "admin"
                  ? "Team Performance Metrics"
                  : "Your Performance Metrics"}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead>Total Deals</TableHead>
                  <TableHead>Won Deals</TableHead>
                  <TableHead>Deal Value</TableHead>
                  <TableHead>Activities</TableHead>
                  <TableHead>Conversion Rate</TableHead>
                  <TableHead>Leads Generated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {userMetrics.map((metric) => (
                  <TableRow key={metric.userId}>
                    <TableCell className="font-medium">
                      {metric.userName}
                      {metric.userId === user?.id && (
                        <Badge variant="outline" className="ml-2 text-xs">
                          You
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>{metric.totalDeals}</TableCell>
                    <TableCell>{metric.wonDeals}</TableCell>
                    <TableCell>${metric.dealValue.toLocaleString()}</TableCell>
                    <TableCell>{metric.activities}</TableCell>
                    <TableCell>{metric.conversionRate}%</TableCell>
                    <TableCell>{metric.leadsGenerated}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Quick Export All */}
      {user?.role === "admin" && (
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Button
                onClick={() => {
                  // Export consolidated report
                  const consolidatedReport: ReportData = {
                    id: "consolidated",
                    name: "Consolidated Company Report",
                    type: "performance",
                    generatedAt: new Date().toISOString(),
                    period: selectedPeriod,
                    status: "completed",
                    data: {
                      totalRevenue: userMetrics.reduce(
                        (sum, u) => sum + u.dealValue,
                        0,
                      ),
                      totalDeals: userMetrics.reduce(
                        (sum, u) => sum + u.totalDeals,
                        0,
                      ),
                      totalActivities: userMetrics.reduce(
                        (sum, u) => sum + u.activities,
                        0,
                      ),
                      avgConversionRate:
                        userMetrics.reduce(
                          (sum, u) => sum + u.conversionRate,
                          0,
                        ) / userMetrics.length,
                    },
                  };
                  exportToExcel(consolidatedReport);
                }}
                variant="outline"
              >
                <Building2 className="h-4 w-4 mr-2" />
                Export Company Report
              </Button>
              <Button
                onClick={() => {
                  // Export all user metrics
                  const userReport: ReportData = {
                    id: "all-users",
                    name: "All Users Performance Report",
                    type: "performance",
                    generatedAt: new Date().toISOString(),
                    period: selectedPeriod,
                    status: "completed",
                    data: {},
                  };
                  exportToExcel(userReport);
                }}
                variant="outline"
              >
                <Users className="h-4 w-4 mr-2" />
                Export Team Report
              </Button>
              <Button
                onClick={() => {
                  // Export all reports as one file
                  filteredReports.forEach((report) => {
                    setTimeout(() => exportToExcel(report), 500);
                  });
                }}
                variant="outline"
              >
                <Download className="h-4 w-4 mr-2" />
                Export All Reports
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
