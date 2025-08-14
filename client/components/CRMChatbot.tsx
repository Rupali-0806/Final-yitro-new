import React, { useState, useRef, useEffect } from "react";
import { useCRM } from "../contexts/CRMContext";
import { useAuth } from "./RealAuthProvider";
import { ChatbotSearchEngine } from "./ChatbotSearchHelpers";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import {
  MessageCircle,
  Send,
  Bot,
  User,
  Minimize2,
  Maximize2,
  X,
  TrendingUp,
  Target,
  Building2,
  Users,
  Calendar,
  DollarSign,
  RotateCcw,
} from "lucide-react";

interface Message {
  id: string;
  content: string;
  sender: "user" | "bot";
  timestamp: Date;
  data?: any;
  quickActions?: string[];
}

interface ChatbotAnalysis {
  topLeads?: any[];
  topAccounts?: any[];
  upcomingDeals?: any[];
  metrics?: any;
  suggestions?: string[];
}

export function CRMChatbot() {
  const { leads, accounts, contacts, deals } = useCRM();
  const { user } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: `Hello ${user?.displayName || "there"}! I'm your intelligent CRM assistant.

I can help you analyze your sales data, track performance, and provide insights about your:

<b>Leads</b> (${leads.length} total)
<b>Accounts</b> (${accounts.length} total)
<b>Deals</b> (${deals.length} total)
<b>Contacts</b> (${contacts.length} total)

<b>Try asking me:</b>
• "Show me top leads this week"
• "What deals are closing soon?"
• "My performance analytics"
• "Show my profile"
• "Search for [company/contact]"

I'm here to help you stay on top of your sales game!`,
      sender: "bot",
      timestamp: new Date(),
      quickActions: [
        "Top leads this week",
        "Deals closing soon",
        "My performance",
        "Account summary",
      ],
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [conversationContext, setConversationContext] = useState<string[]>([]);
  const messageCounterRef = useRef<number>(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const analyzeCRMData = (query: string): ChatbotAnalysis => {
    const lowercaseQuery = query.toLowerCase();

    // Get current date for filtering
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const nextWeek = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);

    // Analyze leads
    if (
      lowercaseQuery.includes("lead") ||
      lowercaseQuery.includes("top lead")
    ) {
      const topLeads = leads.sort((a, b) => b.score - a.score).slice(0, 5);

      const thisWeekLeads = leads.filter((lead) => {
        // Since we don't have exact creation dates, we'll use status for this week
        return lead.status === "New" || lead.status === "Qualified";
      });

      return {
        topLeads,
        metrics: {
          totalLeads: leads.length,
          newLeads: leads.filter((l) => l.status === "New").length,
          qualifiedLeads: leads.filter((l) => l.status === "Qualified").length,
          workingLeads: leads.filter((l) => l.status === "Working").length,
        },
      };
    }

    // Analyze accounts
    if (
      lowercaseQuery.includes("account") ||
      lowercaseQuery.includes("client")
    ) {
      const topAccounts = accounts
        .filter((account) => account.type === "Customer")
        .sort((a, b) => b.activeDeals - a.activeDeals)
        .slice(0, 5);

      return {
        topAccounts,
        metrics: {
          totalAccounts: accounts.length,
          customers: accounts.filter((a) => a.type === "Customer").length,
          prospects: accounts.filter((a) => a.type === "Prospect").length,
          partners: accounts.filter((a) => a.type === "Partner").length,
        },
      };
    }

    // Analyze deals
    if (
      lowercaseQuery.includes("deal") ||
      lowercaseQuery.includes("closing") ||
      lowercaseQuery.includes("pipeline")
    ) {
      const upcomingDeals = deals
        .filter((deal) => {
          const closingDate = new Date(deal.closingDate);
          return (
            closingDate >= now &&
            closingDate <= nextWeek &&
            !["Order Won", "Order Lost"].includes(deal.stage)
          );
        })
        .sort(
          (a, b) =>
            new Date(a.closingDate).getTime() -
            new Date(b.closingDate).getTime(),
        );

      const activeDeals = deals.filter(
        (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
      );
      const totalPipelineValue = activeDeals.reduce(
        (sum, deal) => sum + deal.dealValue,
        0,
      );
      const wonDeals = deals.filter((deal) => deal.stage === "Order Won");
      const totalRevenue = wonDeals.reduce(
        (sum, deal) => sum + deal.dealValue,
        0,
      );

      return {
        upcomingDeals,
        metrics: {
          activeDeals: activeDeals.length,
          totalPipelineValue,
          wonDeals: wonDeals.length,
          totalRevenue,
          avgDealSize:
            activeDeals.length > 0
              ? totalPipelineValue / activeDeals.length
              : 0,
        },
      };
    }

    // Analyze contacts
    if (lowercaseQuery.includes("contact")) {
      const recentContacts = contacts
        .sort(
          (a, b) =>
            new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
        )
        .slice(0, 5);

      return {
        metrics: {
          totalContacts: contacts.length,
          activeDeals: contacts.filter((c) => c.status === "Active Deal")
            .length,
          prospects: contacts.filter((c) => c.status === "Prospect").length,
          suspects: contacts.filter((c) => c.status === "Suspect").length,
        },
      };
    }

    // General overview
    return {
      metrics: {
        totalLeads: leads.length,
        totalAccounts: accounts.length,
        totalContacts: contacts.length,
        totalDeals: deals.length,
        activeDeals: deals.filter(
          (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
        ).length,
      },
    };
  };

  const generateResponse = (query: string): string => {
    const analysis = analyzeCRMData(query);
    const lowercaseQuery = query.toLowerCase();

    // Update conversation context
    setConversationContext((prev) => [...prev.slice(-4), lowercaseQuery]);

    // Handle search queries first
    const searchQuery = ChatbotSearchEngine.extractSearchQuery(query);
    if (searchQuery) {
      const searchResults = ChatbotSearchEngine.performGlobalSearch(
        searchQuery,
        leads,
        accounts,
        contacts,
        deals,
      );
      return ChatbotSearchEngine.formatSearchResults(searchResults);
    }

    // Handle contextual follow-up questions
    if (
      (lowercaseQuery.includes("more") ||
        lowercaseQuery.includes("details") ||
        lowercaseQuery.includes("tell me more")) &&
      conversationContext.length > 0
    ) {
      const lastContext = conversationContext[conversationContext.length - 1];
      if (lastContext.includes("lead")) {
        return generateResponse("show me detailed lead analysis");
      } else if (lastContext.includes("deal")) {
        return generateResponse("detailed pipeline analysis");
      } else if (lastContext.includes("account")) {
        return generateResponse("detailed account breakdown");
      }
    }

    // Handle specific lead inquiries
    if (
      lowercaseQuery.includes("lead") &&
      (lowercaseQuery.includes("week") || lowercaseQuery.includes("this week"))
    ) {
      const topWeeklyLeads = leads
        .filter((lead) => lead.status === "New" || lead.status === "Qualified")
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);

      let response = "<b>Top Leads This Week:</b>\n\n";

      if (topWeeklyLeads.length > 0) {
        topWeeklyLeads.forEach((lead, index) => {
          response += `${index + 1}. <b>${lead.name}</b> from ${lead.company}\n`;
          response += `   Lead Score: ${lead.score}/100\n`;
          response += `   Potential Value: ${lead.value}\n`;
          response += `   Contact: ${lead.phone}\n`;
          response += `   Email: ${lead.email}\n`;
          response += `   Status: ${lead.status}\n\n`;
        });

        response +=
          "<b>Recommendation:</b> Focus on the highest scoring leads first. ";
        response += `${topWeeklyLeads[0]?.name} has the highest score (${topWeeklyLeads[0]?.score}) and should be your priority!`;
      } else {
        response +=
          "No new leads this week. Consider increasing marketing efforts or lead generation activities.";
      }

      return response;
    }

    if (
      lowercaseQuery.includes("lead") ||
      lowercaseQuery.includes("top lead")
    ) {
      let response = "Here are your top leads by score:\n\n";

      if (analysis.topLeads && analysis.topLeads.length > 0) {
        analysis.topLeads.forEach((lead, index) => {
          response += `${index + 1}. <b>${lead.name}</b> from ${lead.company}\n`;
          response += `   • Score: ${lead.score}/100\n`;
          response += `   • Value: ${lead.value}\n`;
          response += `   • Status: ${lead.status}\n`;
          response += `   • Last Activity: ${lead.lastActivity}\n\n`;
        });
      }

      if (analysis.metrics) {
        response += `<b>Lead Summary:</b>\n`;
        response += `• Total Leads: ${analysis.metrics.totalLeads}\n`;
        response += `• New Leads: ${analysis.metrics.newLeads}\n`;
        response += `• Qualified Leads: ${analysis.metrics.qualifiedLeads}\n`;
        response += `• Working Leads: ${analysis.metrics.workingLeads}`;
      }

      return response;
    }

    if (
      lowercaseQuery.includes("account") ||
      lowercaseQuery.includes("client")
    ) {
      let response = "Here are your top accounts:\n\n";

      if (analysis.topAccounts && analysis.topAccounts.length > 0) {
        analysis.topAccounts.forEach((account, index) => {
          response += `${index + 1}. <b>${account.name}</b>\n`;
          response += `   • Industry: ${account.industry}\n`;
          response += `   • Revenue: ${account.revenue}\n`;
          response += `   • Active Deals: ${account.activeDeals}\n`;
          response += `   • Contacts: ${account.contacts}\n`;
          response += `   • Rating: ${account.rating}\n\n`;
        });
      }

      if (analysis.metrics) {
        response += `<b>Account Summary:</b>\n`;
        response += `• Total Accounts: ${analysis.metrics.totalAccounts}\n`;
        response += `• Customers: ${analysis.metrics.customers}\n`;
        response += `• Prospects: ${analysis.metrics.prospects}\n`;
        response += `• Partners: ${analysis.metrics.partners}`;
      }

      return response;
    }

    if (
      lowercaseQuery.includes("deal") ||
      lowercaseQuery.includes("closing") ||
      lowercaseQuery.includes("pipeline")
    ) {
      let response = "";

      if (analysis.upcomingDeals && analysis.upcomingDeals.length > 0) {
        response += "<b>Deals Closing This Week:</b>\n\n";
        analysis.upcomingDeals.forEach((deal, index) => {
          const urgencyLabel =
            deal.probability > 75
              ? "HIGH PRIORITY"
              : deal.probability > 50
                ? "MEDIUM PRIORITY"
                : "LOW PRIORITY";
          response += `${urgencyLabel} ${index + 1}. <b>${deal.dealName}</b>\n`;
          response += `   Account: ${deal.associatedAccount}\n`;
          response += `   Value: $${deal.dealValue.toLocaleString()}\n`;
          response += `   Closing: ${new Date(deal.closingDate).toLocaleDateString()}\n`;
          response += `   Probability: ${deal.probability}%\n`;
          response += `   Stage: ${deal.stage}\n`;
          response += `   Next Step: ${deal.nextStep}\n\n`;
        });

        // Add recommendations based on deal analysis
        const highProbDeals = analysis.upcomingDeals.filter(
          (d) => d.probability > 75,
        );
        if (highProbDeals.length > 0) {
          response += `<b>Action Required:</b> You have ${highProbDeals.length} high-probability deal(s) closing soon. `;
          response += `Focus on "${highProbDeals[0].dealName}" - it's your most likely to close!\n\n`;
        }
      } else {
        response += "No deals are closing this week.\n\n";
        response +=
          "**Suggestion:** Focus on moving deals in your pipeline to the closing stage.\n\n";
      }

      if (analysis.metrics) {
        response += `**Pipeline Summary:**\n`;
        response += `• Active Deals: ${analysis.metrics.activeDeals}\n`;
        response += `• Pipeline Value: $${analysis.metrics.totalPipelineValue.toLocaleString()}\n`;
        response += `• Won Deals: ${analysis.metrics.wonDeals}\n`;
        response += `• Total Revenue: $${analysis.metrics.totalRevenue.toLocaleString()}\n`;
        response += `• Average Deal Size: $${Math.round(analysis.metrics.avgDealSize).toLocaleString()}\n\n`;

        // Add performance insights
        const winRate =
          (analysis.metrics.wonDeals /
            (analysis.metrics.wonDeals + analysis.metrics.activeDeals)) *
          100;
        response += `**Win Rate:** ${Math.round(winRate)}%`;
      }

      return response;
    }

    if (lowercaseQuery.includes("contact")) {
      let response = "Here's your contact summary:\n\n";

      if (analysis.metrics) {
        response += `**Contact Summary:**\n`;
        response += `• Total Contacts: ${analysis.metrics.totalContacts}\n`;
        response += `• Active Deals: ${analysis.metrics.activeDeals}\n`;
        response += `• Prospects: ${analysis.metrics.prospects}\n`;
        response += `• Suspects: ${analysis.metrics.suspects}\n\n`;

        response +=
          "Recent contacts include key decision makers from your top accounts. ";
        response +=
          "Would you like me to show you specific contact details for any account?";
      }

      return response;
    }

    if (
      lowercaseQuery.includes("summary") ||
      lowercaseQuery.includes("overview") ||
      lowercaseQuery.includes("dashboard")
    ) {
      let response = "Here's your CRM overview:\n\n";

      if (analysis.metrics) {
        response += `**Quick Stats:**\n`;
        response += `• Total Leads: ${analysis.metrics.totalLeads}\n`;
        response += `• Total Accounts: ${analysis.metrics.totalAccounts}\n`;
        response += `• Total Contacts: ${analysis.metrics.totalContacts}\n`;
        response += `• Active Deals: ${analysis.metrics.activeDeals}\n\n`;

        response += "Your CRM is looking healthy! ";
        response += "Would you like me to dive deeper into any specific area?";
      }

      return response;
    }

    // Handle user profile queries
    if (
      lowercaseQuery.includes("my profile") ||
      lowercaseQuery.includes("about me") ||
      lowercaseQuery.includes("my info")
    ) {
      let response = `**Your Profile Information:**\n\n`;
      response += `• Name: ${user?.displayName || "Not set"}\n`;
      response += `• Email: ${user?.email || "Not set"}\n`;
      response += `• Role: ${user?.role || "User"}\n`;
      response += `• Account Type: ${user?.role === "admin" ? "Administrator" : "CRM User"}\n\n`;

      response += `**Your CRM Activity:**\n`;
      response += `• Managing ${leads.length} leads\n`;
      response += `• Overseeing ${accounts.length} accounts\n`;
      response += `• Tracking ${deals.length} deals\n`;
      response += `• Connected to ${contacts.length} contacts\n\n`;

      const activeDealsCount = deals.filter(
        (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
      ).length;
      const pipelineValue = deals
        .filter((deal) => !["Order Won", "Order Lost"].includes(deal.stage))
        .reduce((sum, deal) => sum + deal.dealValue, 0);

      response += `**Your Performance:**\n`;
      response += `• Active Deals: ${activeDealsCount}\n`;
      response += `• Pipeline Value: $${pipelineValue.toLocaleString()}\n`;
      response += `• Closed Deals: ${deals.filter((d) => d.stage === "Order Won").length}\n`;

      return response;
    }

    // Handle specific search queries
    if (
      (lowercaseQuery.includes("search") || lowercaseQuery.includes("find")) &&
      !searchQuery
    ) {
      let response = "**Search Help:**\n\n";
      response += "I can help you find specific information. Try asking:\n\n";
      response += '• "Find contact John Smith"\n';
      response += '• "Search for TechCorp account"\n';
      response += '• "Show me deals over $100k"\n';
      response += '• "Find leads from StartupCorp"\n\n';
      response += "What specifically are you looking for?";

      return response;
    }

    // Handle greetings and casual conversation
    if (
      lowercaseQuery.includes("hello") ||
      lowercaseQuery.includes("hi") ||
      lowercaseQuery.includes("hey")
    ) {
      return `Hi there! I'm your CRM assistant. I've got all your latest data ready. What would you like to know about your sales pipeline today?`;
    }

    if (lowercaseQuery.includes("thank") || lowercaseQuery.includes("thanks")) {
      return `You're welcome! I'm always here to help with your CRM data. Is there anything else you'd like to know?`;
    }

    if (lowercaseQuery.includes("help") || lowercaseQuery === "?") {
      return `**I can help you with:**

**Lead Management:** "Top leads this week", "New leads", "Lead status"
**Account Insights:** "Best accounts", "Account summary", "Customer analysis"
**Deal Tracking:** "Closing deals", "Pipeline status", "Deal performance"
**Contact Info:** "Recent contacts", "Find contact [name]"
**Analytics:** "Performance metrics", "Revenue analysis", "My statistics"
**Search:** "Find [anything]", "Search for [company/person]"
**Profile:** "My profile", "My performance", "Account info"

**Pro Tips:**
• I remember our conversation context
• Try "tell me more" for deeper insights
• Use "clear" to reset our conversation

Just ask me naturally - I understand context!`;
    }

    // Handle conversation management
    if (lowercaseQuery.includes("clear") || lowercaseQuery.includes("reset")) {
      setConversationContext([]);
      return "Conversation cleared! How can I help you with your CRM data?";
    }

    // Handle fun easter eggs
    if (lowercaseQuery.includes("joke") || lowercaseQuery.includes("funny")) {
      const jokes = [
        "Why don't sales reps ever get lost? Because they always follow the pipeline!",
        "What do you call a lead that never converts? A cold call forever!",
        "Why did the CRM break up with the spreadsheet? Too many cells, not enough chemistry!",
      ];
      return jokes[Math.floor(Math.random() * jokes.length)];
    }

    if (lowercaseQuery.includes("coffee") || lowercaseQuery.includes("tired")) {
      return "I don't drink coffee, but I can definitely energize you with some exciting sales insights! How about checking your top performing leads?";
    }

    // Handle performance and analytics queries
    if (
      lowercaseQuery.includes("performance") ||
      lowercaseQuery.includes("analytics") ||
      lowercaseQuery.includes("metrics")
    ) {
      const totalRevenue = deals
        .filter((d) => d.stage === "Order Won")
        .reduce((sum, d) => sum + d.dealValue, 0);
      const wonDeals = deals.filter((d) => d.stage === "Order Won");
      const avgDealSize =
        wonDeals.length > 0 ? totalRevenue / wonDeals.length : 0;
      const conversionRate = (wonDeals.length / leads.length) * 100;
      const activeDeals = deals.filter(
        (d) => !["Order Won", "Order Lost"].includes(d.stage),
      );
      const pipelineValue = activeDeals.reduce(
        (sum, d) => sum + d.dealValue,
        0,
      );

      let response = "**Performance Analytics:**\n\n";
      response += `**Revenue Metrics:**\n`;
      response += `• Total Revenue: $${totalRevenue.toLocaleString()}\n`;
      response += `• Pipeline Value: $${pipelineValue.toLocaleString()}\n`;
      response += `• Average Deal Size: $${Math.round(avgDealSize).toLocaleString()}\n`;
      response += `• Lead-to-Deal Conversion: ${Math.round(conversionRate)}%\n\n`;

      response += `**Activity Summary:**\n`;
      response += `• Leads in Pipeline: ${leads.length}\n`;
      response += `• Active Accounts: ${accounts.filter((a) => a.type === "Customer").length}\n`;
      response += `• Deals in Progress: ${activeDeals.length}\n`;
      response += `• Won Deals: ${wonDeals.length}\n\n`;

      // Smart recommendations based on data analysis
      const recommendations = [];

      if (conversionRate < 10) {
        recommendations.push(
          "Focus on lead qualification - your conversion rate needs improvement",
        );
      } else if (conversionRate > 20) {
        recommendations.push(
          "Excellent conversion rate! Consider scaling your lead generation",
        );
      }

      if (activeDeals.length > wonDeals.length * 2) {
        recommendations.push(
          "You have many active deals - focus on closing them",
        );
      }

      if (leads.filter((l) => l.status === "New").length > leads.length * 0.5) {
        recommendations.push(
          "Many new leads need follow-up - prioritize outreach",
        );
      }

      const highValueDeals = activeDeals.filter(
        (d) => d.dealValue > avgDealSize * 1.5,
      );
      if (highValueDeals.length > 0) {
        recommendations.push(
          `Focus on ${highValueDeals.length} high-value deals for maximum impact`,
        );
      }

      if (recommendations.length > 0) {
        response += "**Smart Recommendations:**\n";
        recommendations.forEach((rec) => (response += `• ${rec}\n`));
      } else {
        response +=
          "**Great job!** Your pipeline looks healthy and well-balanced!";
      }

      return response;
    }

    // Default response
    return `I understand you're asking about "${query}". I can help you with information about:

**Leads** - "Show me top leads this week" or "lead status"
**Accounts** - "Show me best accounts" or "account summary"
**Deals** - "What deals are closing?" or "pipeline status"
**Contacts** - "Contact summary" or "recent contacts"
**Overview** - "Dashboard summary" or "CRM overview"
**Profile** - "My profile" or "my performance"
**Analytics** - "Show performance metrics" or "analytics"

What would you like to know more about?`;
  };

  const getSuggestedActions = (query: string): string[] => {
    const lowercaseQuery = query.toLowerCase();

    if (lowercaseQuery.includes("lead")) {
      return ["Account summary", "Pipeline status", "My performance"];
    }

    if (lowercaseQuery.includes("deal") || lowercaseQuery.includes("closing")) {
      return ["Top leads this week", "Account summary", "Revenue analysis"];
    }

    if (lowercaseQuery.includes("account")) {
      return ["Contact summary", "Deals closing soon", "Lead status"];
    }

    if (
      lowercaseQuery.includes("performance") ||
      lowercaseQuery.includes("analytics")
    ) {
      return ["Top leads this week", "Deals closing soon", "Account summary"];
    }

    if (lowercaseQuery.includes("search") || lowercaseQuery.includes("find")) {
      return ["Dashboard summary", "My performance", "Account summary"];
    }

    // Default suggestions
    return ["My performance", "Deals closing soon", "Top leads this week"];
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: `user-${Date.now()}-${++messageCounterRef.current}`,
      content: inputValue,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    // Simulate typing delay
    setTimeout(() => {
      const botResponse = generateResponse(inputValue);
      const suggestedActions = getSuggestedActions(inputValue);
      const botMessage: Message = {
        id: `bot-${Date.now()}-${++messageCounterRef.current}`,
        content: botResponse,
        sender: "bot",
        timestamp: new Date(),
        quickActions: suggestedActions,
      };

      setMessages((prev) => [...prev, botMessage]);
      setIsTyping(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg bg-blue-600 hover:bg-blue-700 z-50"
        size="icon"
      >
        <MessageCircle className="h-6 w-6 text-white" />
      </Button>
    );
  }

  return (
    <Card
      className={`fixed bottom-6 right-6 z-50 shadow-xl transition-all duration-300 ${
        isMinimized ? "w-80 h-16" : "w-96 h-[600px]"
      }`}
    >
      <CardHeader className="pb-3 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
              <Bot className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            </div>
            <CardTitle className="text-lg">CRM Assistant</CardTitle>
          </div>
          <div className="flex items-center space-x-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setMessages([
                  {
                    id: "welcome-reset",
                    content: `Fresh start! I'm ready to help you with your CRM data again.`,
                    sender: "bot",
                    timestamp: new Date(),
                    quickActions: [
                      "Top leads this week",
                      "Deals closing soon",
                      "My performance",
                      "Account summary",
                    ],
                  },
                ]);
                setConversationContext([]);
              }}
              title="Clear conversation"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMinimized(!isMinimized)}
            >
              {isMinimized ? (
                <Maximize2 className="h-4 w-4" />
              ) : (
                <Minimize2 className="h-4 w-4" />
              )}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setIsOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      {!isMinimized && (
        <>
          <CardContent className="flex-1 overflow-hidden p-0">
            <div className="h-[450px] overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`flex items-start space-x-2 max-w-[80%] ${
                      message.sender === "user" ? "flex-row-reverse" : ""
                    }`}
                  >
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.sender === "user"
                          ? "bg-blue-600"
                          : "bg-gray-100 dark:bg-gray-800"
                      }`}
                    >
                      {message.sender === "user" ? (
                        <User className="h-3 w-3 text-white" />
                      ) : (
                        <Bot className="h-3 w-3 text-gray-600 dark:text-gray-400" />
                      )}
                    </div>
                    <div
                      className={`rounded-lg px-3 py-2 ${
                        message.sender === "user"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                      }`}
                    >
                      <div className="text-sm whitespace-pre-line">
                        {message.content}
                      </div>
                      <div
                        className={`text-xs mt-1 opacity-70 ${
                          message.sender === "user"
                            ? "text-blue-100"
                            : "text-gray-500"
                        }`}
                      >
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </div>
                      {message.quickActions && message.sender === "bot" && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {message.quickActions.map((action, index) => (
                            <Button
                              key={index}
                              variant="outline"
                              size="sm"
                              className="text-xs h-6 px-2"
                              onClick={() => {
                                const userMessage: Message = {
                                  id: `user-${Date.now()}-${++messageCounterRef.current}`,
                                  content: action,
                                  sender: "user",
                                  timestamp: new Date(),
                                };
                                setMessages((prev) => [...prev, userMessage]);
                                setIsTyping(true);

                                setTimeout(() => {
                                  const botResponse = generateResponse(action);
                                  const suggestedActions =
                                    getSuggestedActions(action);
                                  const botMessage: Message = {
                                    id: `bot-${Date.now()}-${++messageCounterRef.current}`,
                                    content: botResponse,
                                    sender: "bot",
                                    timestamp: new Date(),
                                    quickActions: suggestedActions,
                                  };
                                  setMessages((prev) => [...prev, botMessage]);
                                  setIsTyping(false);
                                }, 1000);
                              }}
                            >
                              {action}
                            </Button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-2">
                    <div className="w-6 h-6 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center">
                      <Bot className="h-3 w-3 text-gray-600 dark:text-gray-400" />
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-800 rounded-lg px-3 py-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </CardContent>

          <div className="border-t p-4">
            <div className="flex space-x-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me about your CRM data..."
                className="flex-1"
                disabled={isTyping}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isTyping}
                size="sm"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}
