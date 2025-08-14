import React, { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { ScrollArea } from "./ui/scroll-area";
import { Separator } from "./ui/separator";
import {
  Bell,
  Calendar,
  Phone,
  Mail,
  Users,
  DollarSign,
  Clock,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Target,
  MessageSquare,
  Bot,
  FileText,
  Presentation,
  Percent,
  Eye,
} from "lucide-react";
import { useCRM } from "../contexts/CRMContext";
import { format, isToday, isTomorrow, addDays, parseISO } from "date-fns";
import { aiRecommendationService, AIRecommendation } from "../services/aiRecommendationService";

interface Recommendation {
  id: string;
  type:
    | "call"
    | "meeting"
    | "follow-up"
    | "deadline"
    | "opportunity"
    | "urgent"
    | "ai-recommendation";
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
  dueDate?: Date;
  relatedEntity: {
    type: "lead" | "account" | "contact" | "deal";
    id: string;
    name: string;
  };
  action?: {
    label: string;
    onClick: () => void;
  };
  aiRecommendation?: AIRecommendation;
}

export function RecommendationNotifications() {
  const { leads, accounts, contacts, deals } = useCRM();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    generateRecommendations();
  }, [leads, accounts, contacts, deals]);

  const generateRecommendations = () => {
    const newRecommendations: Recommendation[] = [];
    const now = new Date();
    const tomorrow = addDays(now, 1);
    const nextWeek = addDays(now, 7);

    // 1. High-priority leads that need immediate follow-up
    leads
      .filter((lead) => lead.status === "New" && lead.score >= 80)
      .forEach((lead) => {
        newRecommendations.push({
          id: `lead-followup-${lead.id}`,
          type: "call",
          title: "Call High-Score Lead Today",
          description: `${lead.name} from ${lead.company} has a score of ${lead.score}. Strike while hot!`,
          priority: "high",
          dueDate: now,
          relatedEntity: {
            type: "lead",
            id: lead.id.toString(),
            name: lead.name,
          },
          action: {
            label: "Call Now",
            onClick: () => {
              if (lead.phone) {
                // Try to open the phone dialer on mobile devices
                const phoneNumber = lead.phone.replace(/[^\d]/g, '');
                window.open(`tel:${phoneNumber}`, '_self');
              } else {
                alert(`No phone number available for ${lead.name}`);
              }
            },
          },
        });
      });

    // 2. Deals closing this week with high probability
    deals
      .filter((deal) => {
        const closingDate = new Date(deal.closingDate);
        return (
          closingDate >= now &&
          closingDate <= nextWeek &&
          deal.probability >= 70 &&
          !["Order Won", "Order Lost"].includes(deal.stage)
        );
      })
      .forEach((deal) => {
        const closingDate = new Date(deal.closingDate);
        const isUrgent = closingDate <= tomorrow;

        newRecommendations.push({
          id: `deal-closing-${deal.id}`,
          type: "deadline",
          title: isUrgent
            ? "URGENT: Deal Closing Tomorrow!"
            : "Deal Closing This Week",
          description: `${deal.dealName} (${deal.probability}% probability, $${deal.dealValue.toLocaleString()}) - ${deal.nextStep}`,
          priority: isUrgent ? "high" : "medium",
          dueDate: closingDate,
          relatedEntity: {
            type: "deal",
            id: deal.id.toString(),
            name: deal.dealName,
          },
          action: {
            label: "Review Deal",
            onClick: () => console.log(`Reviewing deal ${deal.dealName}`),
          },
        });
      });

    // 3. Qualified leads that haven't been contacted recently
    leads
      .filter(
        (lead) =>
          lead.status === "Qualified" &&
          lead.lastActivity &&
          new Date(lead.lastActivity).getTime() < addDays(now, -3).getTime(),
      )
      .slice(0, 3)
      .forEach((lead) => {
        newRecommendations.push({
          id: `lead-stale-${lead.id}`,
          type: "follow-up",
          title: "Follow-up Required",
          description: `${lead.name} hasn't been contacted in 3+ days. Don't let this qualified lead go cold!`,
          priority: "medium",
          dueDate: now,
          relatedEntity: {
            type: "lead",
            id: lead.id.toString(),
            name: lead.name,
          },
          action: {
            label: "Schedule Call",
            onClick: () => {
              if (lead.phone) {
                const phoneNumber = lead.phone.replace(/[^\d]/g, '');
                window.open(`tel:${phoneNumber}`, '_self');
              } else {
                alert(`No phone number available for ${lead.name}`);
              }
            },
          },
        });
      });

    // 4. Deals in negotiation stage that need attention
    deals
      .filter(
        (deal) =>
          deal.stage === "Negotiating" &&
          new Date(deal.updatedAt).getTime() < addDays(now, -2).getTime(),
      )
      .forEach((deal) => {
        newRecommendations.push({
          id: `deal-negotiation-${deal.id}`,
          type: "urgent",
          title: "Negotiation Needs Attention",
          description: `${deal.dealName} has been in negotiation for 2+ days. Time to push forward!`,
          priority: "high",
          dueDate: now,
          relatedEntity: {
            type: "deal",
            id: deal.id.toString(),
            name: deal.dealName,
          },
          action: {
            label: "Continue Negotiation",
            onClick: () =>
              console.log(`Continuing negotiation for ${deal.dealName}`),
          },
        });
      });

    // 5. High-value accounts without recent activity
    accounts
      .filter((account) => {
        const revenueValue = parseFloat(
          account.revenue?.replace(/[^0-9.-]+/g, "") || "0",
        );
        return (
          account.type === "Customer" &&
          revenueValue > 50000 &&
          account.lastActivity &&
          new Date(account.lastActivity).getTime() < addDays(now, -7).getTime()
        );
      })
      .slice(0, 2)
      .forEach((account) => {
        newRecommendations.push({
          id: `account-check-${account.id}`,
          type: "meeting",
          title: "Check-in with Key Account",
          description: `${account.name} (${account.revenue}) - No activity in 7+ days. Schedule a check-in.`,
          priority: "medium",
          dueDate: tomorrow,
          relatedEntity: {
            type: "account",
            id: account.id.toString(),
            name: account.name,
          },
          action: {
            label: "Schedule Meeting",
            onClick: () =>
              console.log(`Scheduling meeting with ${account.name}`),
          },
        });
      });

    // 6. Opportunities based on deal size vs average
    const avgDealSize =
      deals.length > 0
        ? deals.reduce((sum, deal) => sum + deal.dealValue, 0) / deals.length
        : 0;

    deals
      .filter(
        (deal) =>
          deal.dealValue > avgDealSize * 1.5 &&
          deal.stage === "Proposal Submitted" &&
          !["Order Won", "Order Lost"].includes(deal.stage),
      )
      .slice(0, 2)
      .forEach((deal) => {
        newRecommendations.push({
          id: `deal-opportunity-${deal.id}`,
          type: "opportunity",
          title: "High-Value Opportunity",
          description: `${deal.dealName} ($${deal.dealValue.toLocaleString()}) - 50% above average deal size. Priority focus!`,
          priority: "high",
          dueDate: tomorrow,
          relatedEntity: {
            type: "deal",
            id: deal.id.toString(),
            name: deal.dealName,
          },
          action: {
            label: "Prioritize Deal",
            onClick: () => console.log(`Prioritizing deal ${deal.dealName}`),
          },
        });
      });

    // Sort by priority and due date
    newRecommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const priorityDiff =
        priorityOrder[b.priority] - priorityOrder[a.priority];

      if (priorityDiff !== 0) return priorityDiff;

      if (a.dueDate && b.dueDate) {
        return a.dueDate.getTime() - b.dueDate.getTime();
      }

      return 0;
    });

    setRecommendations(newRecommendations);
    setUnreadCount(newRecommendations.length);
  };

  const getIcon = (type: Recommendation["type"]) => {
    switch (type) {
      case "call":
        return Phone;
      case "meeting":
        return Calendar;
      case "follow-up":
        return MessageSquare;
      case "deadline":
        return Clock;
      case "opportunity":
        return TrendingUp;
      case "urgent":
        return AlertTriangle;
      default:
        return Bell;
    }
  };

  const getPriorityColor = (priority: Recommendation["priority"]) => {
    switch (priority) {
      case "high":
        return "text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800";
      case "medium":
        return "text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-900/20 dark:border-yellow-800";
      case "low":
        return "text-blue-600 bg-blue-50 border-blue-200 dark:text-blue-400 dark:bg-blue-900/20 dark:border-blue-800";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-900/20 dark:border-gray-800";
    }
  };

  const formatDueDate = (date: Date) => {
    if (isToday(date)) return "Today";
    if (isTomorrow(date)) return "Tomorrow";
    return format(date, "MMM d");
  };

  const handleNotificationClick = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setUnreadCount(0); // Mark as read when opened
    }
  };

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="relative hover:bg-gray-100 dark:hover:bg-gray-800"
          onClick={handleNotificationClick}
        >
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <Badge
              variant="destructive"
              className="absolute -top-1 -right-1 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs font-bold"
            >
              {unreadCount > 9 ? "9+" : unreadCount}
            </Badge>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-96 max-h-96 p-0"
        sideOffset={8}
      >
        <Card className="border-0 shadow-none">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center justify-between">
              <span className="flex items-center">
                <Target className="h-4 w-4 mr-2" />
                AI Recommendations
              </span>
              {recommendations.length > 0 && (
                <Badge variant="secondary" className="text-xs">
                  {recommendations.length} items
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {recommendations.length === 0 ? (
              <div className="p-6 text-center">
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-3" />
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
                  All caught up!
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  No urgent recommendations at the moment.
                </p>
              </div>
            ) : (
              <ScrollArea className="h-80">
                <div className="p-2 space-y-2">
                  {recommendations.map((rec, index) => {
                    const Icon = getIcon(rec.type);
                    return (
                      <div key={rec.id}>
                        <div
                          className={`p-3 rounded-lg border cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors ${getPriorityColor(rec.priority)}`}
                          onClick={rec.action?.onClick}
                        >
                          <div className="flex items-start space-x-3">
                            <div className="flex-shrink-0">
                              <Icon className="h-4 w-4 mt-0.5" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between mb-1">
                                <h4 className="text-sm font-medium truncate">
                                  {rec.title}
                                </h4>
                                {rec.dueDate && (
                                  <span className="text-xs opacity-75 ml-2">
                                    {formatDueDate(rec.dueDate)}
                                  </span>
                                )}
                              </div>
                              <p className="text-xs opacity-80 line-clamp-2 mb-2">
                                {rec.description}
                              </p>
                              <div className="flex items-center justify-between">
                                <span className="text-xs opacity-70">
                                  {rec.relatedEntity.type}:{" "}
                                  {rec.relatedEntity.name}
                                </span>
                                {rec.action && (
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    className="h-6 text-xs px-2"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      rec.action?.onClick();
                                    }}
                                  >
                                    {rec.action.label}
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                        {index < recommendations.length - 1 && (
                          <Separator className="my-2" />
                        )}
                      </div>
                    );
                  })}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
