import React, { useState, useRef, useEffect } from "react";
import { useCRM } from "../contexts/CRMContext";
import { useAuth } from "./RealAuthProvider";
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
} from "lucide-react";

interface Message {
  id: string;
  content: string;
  sender: "user" | "bot";
  timestamp: Date;
  data?: any;
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
      content: `Hello ${user?.displayName || "there"}! I'm your CRM assistant. I can help you with information about your leads, accounts, deals, and contacts. Try asking me things like:
      
• "Show me top leads this week"
• "What deals are closing soon?"  
• "Tell me about my best accounts"
• "Show me contact details for [name]"
• "What's my sales pipeline looking like?"`,
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
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
    if (lowercaseQuery.includes("lead") || lowercaseQuery.includes("top lead")) {
      const topLeads = leads
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);
      
      const thisWeekLeads = leads.filter(lead => {
        // Since we don't have exact creation dates, we'll use status for this week
        return lead.status === "New" || lead.status === "Qualified";
      });

      return {
        topLeads,
        metrics: {
          totalLeads: leads.length,
          newLeads: leads.filter(l => l.status === "New").length,
          qualifiedLeads: leads.filter(l => l.status === "Qualified").length,
          workingLeads: leads.filter(l => l.status === "Working").length,
        }
      };
    }

    // Analyze accounts
    if (lowercaseQuery.includes("account") || lowercaseQuery.includes("client")) {
      const topAccounts = accounts
        .filter(account => account.type === "Customer")
        .sort((a, b) => b.activeDeals - a.activeDeals)
        .slice(0, 5);

      return {
        topAccounts,
        metrics: {
          totalAccounts: accounts.length,
          customers: accounts.filter(a => a.type === "Customer").length,
          prospects: accounts.filter(a => a.type === "Prospect").length,
          partners: accounts.filter(a => a.type === "Partner").length,
        }
      };
    }

    // Analyze deals
    if (lowercaseQuery.includes("deal") || lowercaseQuery.includes("closing") || lowercaseQuery.includes("pipeline")) {
      const upcomingDeals = deals
        .filter(deal => {
          const closingDate = new Date(deal.closingDate);
          return closingDate >= now && closingDate <= nextWeek && 
                 !["Order Won", "Order Lost"].includes(deal.stage);
        })
        .sort((a, b) => new Date(a.closingDate).getTime() - new Date(b.closingDate).getTime());

      const activeDeals = deals.filter(deal => !["Order Won", "Order Lost"].includes(deal.stage));
      const totalPipelineValue = activeDeals.reduce((sum, deal) => sum + deal.dealValue, 0);
      const wonDeals = deals.filter(deal => deal.stage === "Order Won");
      const totalRevenue = wonDeals.reduce((sum, deal) => sum + deal.dealValue, 0);

      return {
        upcomingDeals,
        metrics: {
          activeDeals: activeDeals.length,
          totalPipelineValue,
          wonDeals: wonDeals.length,
          totalRevenue,
          avgDealSize: activeDeals.length > 0 ? totalPipelineValue / activeDeals.length : 0,
        }
      };
    }

    // Analyze contacts
    if (lowercaseQuery.includes("contact")) {
      const recentContacts = contacts
        .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
        .slice(0, 5);

      return {
        metrics: {
          totalContacts: contacts.length,
          activeDeals: contacts.filter(c => c.status === "Active Deal").length,
          prospects: contacts.filter(c => c.status === "Prospect").length,
          suspects: contacts.filter(c => c.status === "Suspect").length,
        }
      };
    }

    // General overview
    return {
      metrics: {
        totalLeads: leads.length,
        totalAccounts: accounts.length,
        totalContacts: contacts.length,
        totalDeals: deals.length,
        activeDeals: deals.filter(deal => !["Order Won", "Order Lost"].includes(deal.stage)).length,
      }
    };
  };

  const generateResponse = (query: string): string => {
    const analysis = analyzeCRMData(query);
    const lowercaseQuery = query.toLowerCase();

    if (lowercaseQuery.includes("lead") || lowercaseQuery.includes("top lead")) {
      let response = "Here are your top leads by score:\n\n";
      
      if (analysis.topLeads && analysis.topLeads.length > 0) {
        analysis.topLeads.forEach((lead, index) => {
          response += `${index + 1}. **${lead.name}** from ${lead.company}\n`;
          response += `   • Score: ${lead.score}/100\n`;
          response += `   • Value: ${lead.value}\n`;
          response += `   • Status: ${lead.status}\n`;
          response += `   • Last Activity: ${lead.lastActivity}\n\n`;
        });
      }

      if (analysis.metrics) {
        response += `**Lead Summary:**\n`;
        response += `• Total Leads: ${analysis.metrics.totalLeads}\n`;
        response += `• New Leads: ${analysis.metrics.newLeads}\n`;
        response += `• Qualified Leads: ${analysis.metrics.qualifiedLeads}\n`;
        response += `• Working Leads: ${analysis.metrics.workingLeads}`;
      }

      return response;
    }

    if (lowercaseQuery.includes("account") || lowercaseQuery.includes("client")) {
      let response = "Here are your top accounts:\n\n";
      
      if (analysis.topAccounts && analysis.topAccounts.length > 0) {
        analysis.topAccounts.forEach((account, index) => {
          response += `${index + 1}. **${account.name}**\n`;
          response += `   • Industry: ${account.industry}\n`;
          response += `   • Revenue: ${account.revenue}\n`;
          response += `   • Active Deals: ${account.activeDeals}\n`;
          response += `   • Contacts: ${account.contacts}\n`;
          response += `   • Rating: ${account.rating}\n\n`;
        });
      }

      if (analysis.metrics) {
        response += `**Account Summary:**\n`;
        response += `• Total Accounts: ${analysis.metrics.totalAccounts}\n`;
        response += `• Customers: ${analysis.metrics.customers}\n`;
        response += `• Prospects: ${analysis.metrics.prospects}\n`;
        response += `• Partners: ${analysis.metrics.partners}`;
      }

      return response;
    }

    if (lowercaseQuery.includes("deal") || lowercaseQuery.includes("closing") || lowercaseQuery.includes("pipeline")) {
      let response = "";
      
      if (analysis.upcomingDeals && analysis.upcomingDeals.length > 0) {
        response += "Here are your deals closing this week:\n\n";
        analysis.upcomingDeals.forEach((deal, index) => {
          response += `${index + 1}. **${deal.dealName}**\n`;
          response += `   • Account: ${deal.associatedAccount}\n`;
          response += `   • Value: $${deal.dealValue.toLocaleString()}\n`;
          response += `   • Closing Date: ${new Date(deal.closingDate).toLocaleDateString()}\n`;
          response += `   • Probability: ${deal.probability}%\n`;
          response += `   • Stage: ${deal.stage}\n`;
          response += `   • Next Step: ${deal.nextStep}\n\n`;
        });
      } else {
        response += "No deals are closing this week.\n\n";
      }

      if (analysis.metrics) {
        response += `**Pipeline Summary:**\n`;
        response += `• Active Deals: ${analysis.metrics.activeDeals}\n`;
        response += `• Pipeline Value: $${analysis.metrics.totalPipelineValue.toLocaleString()}\n`;
        response += `• Won Deals: ${analysis.metrics.wonDeals}\n`;
        response += `• Total Revenue: $${analysis.metrics.totalRevenue.toLocaleString()}\n`;
        response += `• Average Deal Size: $${Math.round(analysis.metrics.avgDealSize).toLocaleString()}`;
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
        
        response += "Recent contacts include key decision makers from your top accounts. ";
        response += "Would you like me to show you specific contact details for any account?";
      }

      return response;
    }

    if (lowercaseQuery.includes("summary") || lowercaseQuery.includes("overview") || lowercaseQuery.includes("dashboard")) {
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

    // Default response
    return `I understand you're asking about "${query}". I can help you with information about:

• **Leads** - "Show me top leads" or "lead status"
• **Accounts** - "Show me best accounts" or "account summary" 
• **Deals** - "What deals are closing?" or "pipeline status"
• **Contacts** - "Contact summary" or "recent contacts"
• **Overview** - "Dashboard summary" or "CRM overview"

What would you like to know more about?`;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: inputValue,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    // Simulate typing delay
    setTimeout(() => {
      const botResponse = generateResponse(inputValue);
      const botMessage: Message = {
        id: `bot-${Date.now()}`,
        content: botResponse,
        sender: "bot",
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
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
    <Card className={`fixed bottom-6 right-6 z-50 shadow-xl transition-all duration-300 ${
      isMinimized ? "w-80 h-16" : "w-96 h-[600px]"
    }`}>
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
              onClick={() => setIsMinimized(!isMinimized)}
            >
              {isMinimized ? (
                <Maximize2 className="h-4 w-4" />
              ) : (
                <Minimize2 className="h-4 w-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(false)}
            >
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
                          message.sender === "user" ? "text-blue-100" : "text-gray-500"
                        }`}
                      >
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </div>
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
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
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
