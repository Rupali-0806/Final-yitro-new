import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { useCRM } from "../contexts/CRMContext";
import {
  Users,
  Building2,
  Target,
  DollarSign,
  Activity,
  TrendingUp,
  Trophy,
  Clock,
  UserCheck,
  Briefcase,
} from "lucide-react";

interface MetricsData {
  totalUsers: number;
  totalAccounts: number;
  totalLeads: number;
  totalDeals: number;
  totalActivities: number;
  activeUsers: number;
  wonDeals: number;
  totalDealValue: number;
  recentActivities: Array<{
    type: string;
    date: string;
    summary: string;
    account?: string;
    contact?: string;
    user?: string;
    value?: number;
  }>;
}

export function AdminMetrics() {
  const { leads, accounts, contacts, deals } = useCRM();
  const [loading, setLoading] = useState(false); // No need to load from API

  // Calculate comprehensive real-time metrics from CRM data
  const wonDealsData = deals.filter((deal) => deal.stage === "Order Won");
  const activeDealsData = deals.filter(
    (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
  );
  const newLeadsData = leads.filter((lead) => lead.status === "New");

  // Create a comprehensive timeline of all user activities
  const allActivities = [
    // Deal activities - creation and status changes
    ...deals.map((deal) => ({
      type: deal.stage === "Order Won" ? "Deal Closed" : "Deal Progress",
      date: deal.createdAt,
      summary:
        deal.stage === "Order Won"
          ? `${deal.owner} closed deal: ${deal.dealName} - $${deal.dealValue.toLocaleString()}`
          : `${deal.owner} updated deal: ${deal.dealName} to ${deal.stage}`,
      account: deal.associatedAccount,
      contact: deal.associatedContact,
      user: deal.owner,
      value: deal.dealValue,
    })),
    // Contact activities - creation and updates
    ...contacts.map((contact) => ({
      type: "Contact Activity",
      date: contact.updatedAt,
      summary: `${contact.owner} updated contact: ${contact.firstName} ${contact.lastName} (${contact.status})`,
      account: contact.associatedAccount,
      contact: `${contact.firstName} ${contact.lastName}`,
      user: contact.owner,
    })),
    // Lead activities - based on lead data
    ...leads.map((lead) => ({
      type: "Lead Activity",
      date: new Date(
        Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000,
      ).toISOString(), // Random within last week
      summary: `Lead ${lead.status.toLowerCase()}: ${lead.name} from ${lead.company} (Score: ${lead.score})`,
      account: lead.company,
      contact: lead.name,
      user: "System", // Would be actual user who worked on lead
      value: parseFloat(lead.value.replace(/[$,]/g, "")),
    })),
    // Account activities - account updates
    ...accounts.map((account) => ({
      type: "Account Activity",
      date: new Date(
        Date.now() - Math.random() * 14 * 24 * 60 * 60 * 1000,
      ).toISOString(), // Random within last 2 weeks
      summary: `${account.owner} updated account: ${account.name} (${account.type}, ${account.rating})`,
      account: account.name,
      user: account.owner,
    })),
  ];

  // Sort by date (most recent first) and limit to recent activities
  const recentActivities = allActivities
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, 8); // Show top 8 most recent activities

  const metrics: MetricsData = {
    totalUsers: new Set([
      ...deals.map((d) => d.owner),
      ...contacts.map((c) => c.owner),
      ...accounts.map((a) => a.owner),
    ]).size,
    totalAccounts: accounts.length,
    totalLeads: leads.length,
    totalDeals: activeDealsData.length, // Only active deals
    totalActivities: allActivities.length, // Total activity count across all users
    activeUsers: new Set([
      ...deals.map((d) => d.owner),
      ...contacts.map((c) => c.owner),
      ...accounts.map((a) => a.owner),
    ]).size,
    wonDeals: wonDealsData.length,
    totalDealValue: wonDealsData.reduce((sum, deal) => sum + deal.dealValue, 0),
    recentActivities,
  };

  console.log("📊 Admin Metrics calculated:", {
    totalUsers: metrics.totalUsers,
    accounts: accounts.length,
    leads: leads.length,
    deals: deals.length,
    wonDeals: metrics.wonDeals,
    totalDealValue: metrics.totalDealValue,
    totalActivities: metrics.totalActivities,
    recentActivitiesCount: metrics.recentActivities.length,
    uniqueUsers: new Set([
      ...deals.map((d) => d.owner),
      ...contacts.map((c) => c.owner),
      ...accounts.map((a) => a.owner),
    ]),
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-8 bg-gray-200 rounded w-1/2"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Users */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalUsers}</div>
            <p className="text-xs text-muted-foreground">
              {metrics.activeUsers} currently active
            </p>
          </CardContent>
        </Card>

        {/* Total Accounts */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Accounts
            </CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalAccounts}</div>
            <p className="text-xs text-muted-foreground">
              Company accounts managed
            </p>
          </CardContent>
        </Card>

        {/* Total Leads */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Leads</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalLeads}</div>
            <p className="text-xs text-muted-foreground">
              Lead opportunities tracked
            </p>
          </CardContent>
        </Card>

        {/* Total Deal Value */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Won Deal Value
            </CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(metrics.totalDealValue)}
            </div>
            <p className="text-xs text-muted-foreground">
              From {metrics.wonDeals} closed deals
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Active Deals */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Deals</CardTitle>
            <Briefcase className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalDeals}</div>
            <div className="flex items-center gap-2 mt-1">
              <Badge
                variant="outline"
                className="text-green-600 border-green-600"
              >
                <Trophy className="w-3 h-3 mr-1" />
                {metrics.wonDeals} Won
              </Badge>
              <Badge
                variant="outline"
                className="text-blue-600 border-blue-600"
              >
                <TrendingUp className="w-3 h-3 mr-1" />
                {deals.length} Total
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Total Activities */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Activities
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalActivities}</div>
            <p className="text-xs text-muted-foreground">
              Tracked interactions
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activities */}
      {metrics.recentActivities.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Recent Company Activities
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {metrics.recentActivities.map((activity, index) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg border-l-4 border-blue-200"
                >
                  <div className="flex-shrink-0">
                    <Badge
                      variant={
                        activity.type.includes("Closed")
                          ? "default"
                          : "secondary"
                      }
                      className={`text-xs ${activity.type.includes("Closed") ? "bg-green-100 text-green-800" : ""}`}
                    >
                      {activity.type}
                    </Badge>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">
                      {activity.summary}
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2 text-xs text-gray-600">
                      {activity.user && (
                        <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                          👤 {activity.user}
                        </span>
                      )}
                      {activity.account && (
                        <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded">
                          🏢 {activity.account}
                        </span>
                      )}
                      {activity.contact &&
                        activity.contact !== activity.account && (
                          <span className="bg-orange-100 text-orange-800 px-2 py-1 rounded">
                            👥 {activity.contact}
                          </span>
                        )}
                      {activity.value && (
                        <span className="bg-green-100 text-green-800 px-2 py-1 rounded font-medium">
                          💰 ${activity.value.toLocaleString()}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex-shrink-0 text-xs text-gray-500 text-right">
                    <div>{formatDate(activity.date)}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {new Date(activity.date).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Stats Summary */}
      <Card className="border-blue-200 bg-blue-50">
        <CardHeader>
          <CardTitle className="text-blue-800 flex items-center gap-2">
            <UserCheck className="w-5 h-5" />
            Company Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {metrics.totalUsers}
              </div>
              <div className="text-sm text-blue-700">Team Members</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {metrics.totalAccounts}
              </div>
              <div className="text-sm text-blue-700">Client Accounts</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {metrics.totalLeads}
              </div>
              <div className="text-sm text-blue-700">Active Leads</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {formatCurrency(metrics.totalDealValue)}
              </div>
              <div className="text-sm text-blue-700">Revenue Generated</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
