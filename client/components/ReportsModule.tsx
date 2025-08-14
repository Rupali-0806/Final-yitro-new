import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Download,
  FileText,
  BarChart3,
  TrendingUp,
  Calendar,
  Filter,
  RefreshCw,
} from "lucide-react";

interface ReportMetrics {
  meetingsFixed?: number;
  meetingsCompleted?: number;
  opportunityCreatedNos?: number;
  opportunityCreatedValue?: number;
  proposalSubmittedNos?: number;
  proposalSubmittedValue?: number;
  orderWonNos?: number;
  orderWonValue?: number;
  orderLostNos?: number;
  orderLostValue?: number;
  meetingsFixedVsTarget?: number;
  meetingsCompletedVsTarget?: number;
  opportunityCreatedVsTarget?: number;
  opportunityCreatedValueVsTarget?: number;
  proposalSubmittedVsTarget?: number;
  proposalSubmittedValueVsTarget?: number;
  orderWonVsTarget?: number;
  orderWonValueVsTarget?: number;
}

interface ReportsModuleProps {
  onBack: () => void;
}

export default function ReportsModule({ onBack }: ReportsModuleProps) {
  const [reportType, setReportType] = useState("sales-performance");
  const [period, setPeriod] = useState("month");
  const [salesRep, setSalesRep] = useState("all");
  const [geo, setGeo] = useState("all");
  const [businessLine, setBusinessLine] = useState("all");
  const [org, setOrg] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportData, setReportData] = useState<ReportMetrics | null>(null);

  const handleGenerateReport = async () => {
    setIsGenerating(true);

    try {
      const response = await fetch("/api/reports/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          reportType,
          period,
          salesRep: salesRep === "all" ? "" : salesRep,
          geo: geo === "all" ? "" : geo,
          businessLine: businessLine === "all" ? "" : businessLine,
          org,
          format: "json",
        }),
      });

      const result = await response.json();
      if (result.success) {
        setReportData(result.data.metrics);
      }
    } catch (error) {
      console.error("Error generating report:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async (format: "excel" | "pdf") => {
    try {
      const response = await fetch("/api/reports/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          reportType,
          period,
          salesRep: salesRep === "all" ? "" : salesRep,
          geo: geo === "all" ? "" : geo,
          businessLine: businessLine === "all" ? "" : businessLine,
          org,
          format,
        }),
      });

      const result = await response.json();
      if (result.success && result.downloadUrl) {
        // In a real implementation, this would trigger a file download
        window.open(result.downloadUrl, "_blank");
      }
    } catch (error) {
      console.error("Error downloading report:", error);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value}%`;
  };

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="p-6 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
              Reports & Analytics
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Generate and download comprehensive sales reports
            </p>
          </div>
          <Button variant="outline" onClick={onBack}>
            Back to Dashboard
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="p-6 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Report Type
            </label>
            <Select value={reportType} onValueChange={setReportType}>
              <SelectTrigger>
                <SelectValue placeholder="Select report type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sales-performance">
                  Sales Performance
                </SelectItem>
                <SelectItem value="activity-summary">
                  Activity Summary
                </SelectItem>
                <SelectItem value="pipeline-analysis">
                  Pipeline Analysis
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Period
            </label>
            <Select value={period} onValueChange={setPeriod}>
              <SelectTrigger>
                <SelectValue placeholder="Select period" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="month">Monthly</SelectItem>
                <SelectItem value="quarter">Quarterly</SelectItem>
                <SelectItem value="year">Yearly</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Sales Rep
            </label>
            <Select value={salesRep} onValueChange={setSalesRep}>
              <SelectTrigger>
                <SelectValue placeholder="All Sales Reps" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sales Reps</SelectItem>
                <SelectItem value="sales-rep-1">Sales Rep 1</SelectItem>
                <SelectItem value="sales-rep-2">Sales Rep 2</SelectItem>
                <SelectItem value="sales-rep-3">Sales Rep 3</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Geography
            </label>
            <Select value={geo} onValueChange={setGeo}>
              <SelectTrigger>
                <SelectValue placeholder="All Regions" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Regions</SelectItem>
                <SelectItem value="Americas">Americas</SelectItem>
                <SelectItem value="India">India</SelectItem>
                <SelectItem value="Philippines">Philippines</SelectItem>
                <SelectItem value="EMEA">EMEA</SelectItem>
                <SelectItem value="ANZ">ANZ</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Business Line
            </label>
            <Select value={businessLine} onValueChange={setBusinessLine}>
              <SelectTrigger>
                <SelectValue placeholder="All Business Lines" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Business Lines</SelectItem>
                <SelectItem value="Human Capital">Human Capital</SelectItem>
                <SelectItem value="Managed Services">
                  Managed Services
                </SelectItem>
                <SelectItem value="GCC">GCC</SelectItem>
                <SelectItem value="Automation">Automation</SelectItem>
                <SelectItem value="Support">Support</SelectItem>
                <SelectItem value="Product">Product</SelectItem>
                <SelectItem value="Solution">Solution</SelectItem>
                <SelectItem value="RCM">RCM</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Organization
            </label>
            <Input
              placeholder="Enter organization"
              value={org}
              onChange={(e) => setOrg(e.target.value)}
            />
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <Button
            onClick={handleGenerateReport}
            disabled={isGenerating}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            {isGenerating ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4 mr-2" />
                Generate Report
              </>
            )}
          </Button>

          {reportData && (
            <>
              <Button
                onClick={() => handleDownload("excel")}
                variant="outline"
                className="text-green-600 border-green-600 hover:bg-green-50"
              >
                <Download className="h-4 w-4 mr-2" />
                Download Excel
              </Button>

              <Button
                onClick={() => handleDownload("pdf")}
                variant="outline"
                className="text-red-600 border-red-600 hover:bg-red-50"
              >
                <FileText className="h-4 w-4 mr-2" />
                Download PDF
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Report Data */}
      <div className="flex-1 p-6 overflow-auto">
        {reportData ? (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Key Performance Metrics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Meetings Fixed
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-blue-600">
                      {reportData.meetingsFixed}
                    </div>
                    <div className="text-sm text-gray-500">
                      vs Target:{" "}
                      {formatPercentage(reportData.meetingsFixedVsTarget || 0)}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Meetings Completed
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-green-600">
                      {reportData.meetingsCompleted}
                    </div>
                    <div className="text-sm text-gray-500">
                      vs Target:{" "}
                      {formatPercentage(
                        reportData.meetingsCompletedVsTarget || 0,
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Active Deals Created
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-purple-600">
                      {reportData.opportunityCreatedNos}
                    </div>
                    <div className="text-sm text-gray-500">
                      Value:{" "}
                      {formatCurrency(reportData.opportunityCreatedValue || 0)}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-gray-600">
                      Orders Won
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-green-600">
                      {reportData.orderWonNos}
                    </div>
                    <div className="text-sm text-gray-500">
                      Value: {formatCurrency(reportData.orderWonValue || 0)}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Detailed Metrics Table */}
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Detailed Performance Analysis
              </h2>
              <Card>
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-gray-50 dark:bg-gray-800">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Metric
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Actual
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            vs Target (%)
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                            Meetings Fixed
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.meetingsFixed}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.meetingsFixedVsTarget}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                (reportData.meetingsFixedVsTarget || 0) >= 100
                                  ? "bg-green-100 text-green-800"
                                  : "bg-red-100 text-red-800"
                              }`}
                            >
                              {(reportData.meetingsFixedVsTarget || 0) >= 100
                                ? "Above Target"
                                : "Below Target"}
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                            Meetings Completed
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.meetingsCompleted}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.meetingsCompletedVsTarget}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                (reportData.meetingsCompletedVsTarget || 0) >=
                                100
                                  ? "bg-green-100 text-green-800"
                                  : "bg-red-100 text-red-800"
                              }`}
                            >
                              {(reportData.meetingsCompletedVsTarget || 0) >=
                              100
                                ? "Above Target"
                                : "Below Target"}
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                            Active Deals Created
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.opportunityCreatedNos}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.opportunityCreatedVsTarget}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                (reportData.opportunityCreatedVsTarget || 0) >=
                                100
                                  ? "bg-green-100 text-green-800"
                                  : "bg-red-100 text-red-800"
                              }`}
                            >
                              {(reportData.opportunityCreatedVsTarget || 0) >=
                              100
                                ? "Above Target"
                                : "Below Target"}
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                            Proposals Submitted
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.proposalSubmittedNos}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.proposalSubmittedVsTarget}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                (reportData.proposalSubmittedVsTarget || 0) >=
                                100
                                  ? "bg-green-100 text-green-800"
                                  : "bg-red-100 text-red-800"
                              }`}
                            >
                              {(reportData.proposalSubmittedVsTarget || 0) >=
                              100
                                ? "Above Target"
                                : "Below Target"}
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                            Orders Won
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.orderWonNos}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {reportData.orderWonVsTarget}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                (reportData.orderWonVsTarget || 0) >= 100
                                  ? "bg-green-100 text-green-800"
                                  : "bg-red-100 text-red-800"
                              }`}
                            >
                              {(reportData.orderWonVsTarget || 0) >= 100
                                ? "Above Target"
                                : "Below Target"}
                            </span>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              No Report Generated
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Configure your filters and click "Generate Report" to view
              analytics data
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
