import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { useCRM } from "../contexts/CRMContext";
import {
  Building2,
  Users,
  Calendar,
  DollarSign,
  FileText,
  Phone,
  Mail,
  MapPin,
  Globe,
  Target,
  TrendingUp,
  Clock,
  Plus,
  Edit,
  Eye,
} from "lucide-react";

interface Task {
  id: string;
  title: string;
  description: string;
  dueDate: string;
  status: "pending" | "in_progress" | "completed";
  priority: "low" | "medium" | "high";
  type: "call" | "email" | "meeting" | "follow_up";
}

interface Document {
  id: string;
  name: string;
  type: string;
  size: string;
  uploadDate: string;
  category: "contract" | "proposal" | "presentation" | "other";
}

interface Interaction {
  id: string;
  type: "call" | "email" | "meeting" | "note";
  date: string;
  subject: string;
  description: string;
  contact: string;
  outcome: string;
}

export default function ClientView() {
  const { accountId } = useParams<{ accountId: string }>();
  const { accounts, contacts, deals, activities } = useCRM();

  const [account, setAccount] = useState<any>(null);
  const [accountContacts, setAccountContacts] = useState<any[]>([]);
  const [accountDeals, setAccountDeals] = useState<any[]>([]);
  const [accountActivities, setAccountActivities] = useState<any[]>([]);

  // Mock data for features not yet in CRM
  const [tasks, setTasks] = useState<Task[]>([
    {
      id: "1",
      title: "Follow up on proposal",
      description: "Check if client has reviewed the enterprise proposal",
      dueDate: "2024-01-25",
      status: "pending",
      priority: "high",
      type: "call",
    },
    {
      id: "2",
      title: "Schedule product demo",
      description: "Set up technical demo for decision makers",
      dueDate: "2024-01-28",
      status: "in_progress",
      priority: "medium",
      type: "meeting",
    },
  ]);

  const [documents, setDocuments] = useState<Document[]>([
    {
      id: "1",
      name: "Enterprise_Proposal_v2.pdf",
      type: "PDF",
      size: "2.3 MB",
      uploadDate: "2024-01-20",
      category: "proposal",
    },
    {
      id: "2",
      name: "Service_Agreement.docx",
      type: "DOCX",
      size: "845 KB",
      uploadDate: "2024-01-18",
      category: "contract",
    },
  ]);

  const [interactions, setInteractions] = useState<Interaction[]>([
    {
      id: "1",
      type: "call",
      date: "2024-01-22",
      subject: "Discovery call with CTO",
      description: "Discussed technical requirements and integration needs",
      contact: "John Smith",
      outcome: "Positive - interested in enterprise features",
    },
    {
      id: "2",
      type: "email",
      date: "2024-01-20",
      subject: "Proposal sent",
      description: "Sent updated enterprise proposal with custom pricing",
      contact: "Sarah Johnson",
      outcome: "Awaiting review",
    },
  ]);

  useEffect(() => {
    if (accountId && accounts && contacts && deals && activities) {
      // Find account details
      const foundAccount = accounts.find((acc) => acc.id === accountId);
      setAccount(foundAccount);

      // Find related contacts
      const relatedContacts = contacts.filter(
        (contact) => contact.associatedAccount === accountId,
      );
      setAccountContacts(relatedContacts);

      // Find related deals
      const relatedDeals = deals.filter(
        (deal) => deal.associatedAccount === foundAccount?.name,
      );
      setAccountDeals(relatedDeals);

      // Find related activities
      const relatedActivities = activities.filter(
        (activity) => activity.associatedAccount === accountId,
      );
      setAccountActivities(relatedActivities);
    }
  }, [accountId, accounts, contacts, deals, activities]);

  if (!account) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900">
            Account not found
          </h1>
          <p className="text-gray-600 mt-2">
            The requested account could not be found.
          </p>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800";
      case "in_progress":
        return "bg-blue-100 text-blue-800";
      case "pending":
        return "bg-yellow-100 text-yellow-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-orange-100 text-orange-800";
      case "low":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const totalDealValue =
    accountDeals?.reduce((sum, deal) => sum + deal.dealValue, 0) || 0;
  const activeDealCount =
    accountDeals?.filter(
      (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
    ).length || 0;

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
            <Building2 className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{account.name}</h1>
            <p className="text-gray-600">
              {account.industry} • {account.type}
            </p>
          </div>
        </div>
        <Badge variant="secondary" className="px-3 py-1">
          {account.rating} Rating
        </Badge>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-4 w-4 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Total Deal Value</p>
                <p className="text-xl font-bold">
                  ${totalDealValue.toLocaleString()}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-600" />
              <div>
                <p className="text-sm text-gray-600">Active Deals</p>
                <p className="text-xl font-bold">{activeDealCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Users className="h-4 w-4 text-purple-600" />
              <div>
                <p className="text-sm text-gray-600">Contacts</p>
                <p className="text-xl font-bold">
                  {accountContacts?.length || 0}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-orange-600" />
              <div>
                <p className="text-sm text-gray-600">Revenue</p>
                <p className="text-xl font-bold">{account.revenue}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="contacts">Contacts</TabsTrigger>
          <TabsTrigger value="deals">Opportunities</TabsTrigger>
          <TabsTrigger value="tasks">Tasks</TabsTrigger>
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="interactions">Interactions</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Account Information */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Building2 className="h-5 w-5" />
                  <span>Account Information</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Globe className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Industry:</span>
                  <span className="text-sm font-medium">
                    {account.industry}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <MapPin className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Location:</span>
                  <span className="text-sm font-medium">
                    {account.location || "Not specified"}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <DollarSign className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Annual Revenue:</span>
                  <span className="text-sm font-medium">{account.revenue}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Users className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Employees:</span>
                  <span className="text-sm font-medium">
                    {account.employees || "Not specified"}
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Clock className="h-5 w-5" />
                  <span>Recent Activity</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {accountActivities?.slice(0, 5).map((activity, index) => (
                    <div
                      key={index}
                      className="flex items-start space-x-3 pb-3 border-b border-gray-100 last:border-0"
                    >
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <div className="flex-1">
                        <p className="text-sm font-medium">
                          {activity.activityType}
                        </p>
                        <p className="text-xs text-gray-600">
                          {activity.description}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {new Date(activity.dateTime).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  )) || []}
                  {(accountActivities?.length || 0) === 0 && (
                    <p className="text-sm text-gray-500">
                      No recent activities
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Contacts Tab */}
        <TabsContent value="contacts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Contacts ({accountContacts?.length || 0})</span>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  Add Contact
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {accountContacts.map((contact) => (
                  <div
                    key={contact.id}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                        <Users className="h-5 w-5 text-gray-600" />
                      </div>
                      <div>
                        <h3 className="font-medium">
                          {contact.firstName} {contact.lastName}
                        </h3>
                        <p className="text-sm text-gray-600">{contact.title}</p>
                        <div className="flex items-center space-x-4 mt-1">
                          <span className="flex items-center text-sm text-gray-500">
                            <Mail className="h-3 w-3 mr-1" />
                            {contact.emailAddress}
                          </span>
                          <span className="flex items-center text-sm text-gray-500">
                            <Phone className="h-3 w-3 mr-1" />
                            {contact.deskPhone}
                          </span>
                        </div>
                      </div>
                    </div>
                    <Badge
                      variant={
                        contact.status === "Active Deal"
                          ? "default"
                          : "secondary"
                      }
                    >
                      {contact.status}
                    </Badge>
                  </div>
                ))}
                {accountContacts.length === 0 && (
                  <p className="text-center text-gray-500 py-8">
                    No contacts found for this account
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Deals Tab */}
        <TabsContent value="deals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Opportunities ({accountDeals.length})</span>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  New Opportunity
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {accountDeals.map((deal) => (
                  <div key={deal.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-medium">{deal.dealName}</h3>
                      <Badge
                        variant={
                          deal.stage === "Order Won" ? "default" : "secondary"
                        }
                      >
                        {deal.stage}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Value:</span>
                        <p className="font-medium">
                          ${deal.dealValue.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <span className="text-gray-600">Probability:</span>
                        <p className="font-medium">{deal.probability}%</p>
                      </div>
                      <div>
                        <span className="text-gray-600">Closing Date:</span>
                        <p className="font-medium">
                          {new Date(deal.closingDate).toLocaleDateString()}
                        </p>
                      </div>
                      <div>
                        <span className="text-gray-600">Next Step:</span>
                        <p className="font-medium">{deal.nextStep}</p>
                      </div>
                    </div>
                  </div>
                ))}
                {accountDeals.length === 0 && (
                  <p className="text-center text-gray-500 py-8">
                    No opportunities found for this account
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tasks Tab */}
        <TabsContent value="tasks" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Tasks ({tasks.length})</span>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  New Task
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {tasks.map((task) => (
                  <div key={task.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{task.title}</h3>
                      <div className="flex space-x-2">
                        <Badge className={getPriorityColor(task.priority)}>
                          {task.priority}
                        </Badge>
                        <Badge className={getStatusColor(task.status)}>
                          {task.status}
                        </Badge>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">
                      {task.description}
                    </p>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">
                        Due: {new Date(task.dueDate).toLocaleDateString()}
                      </span>
                      <span className="text-gray-500">Type: {task.type}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Documents Tab */}
        <TabsContent value="documents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Documents ({documents.length})</span>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  Upload Document
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                        <FileText className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="font-medium">{doc.name}</h3>
                        <div className="flex items-center space-x-4 text-sm text-gray-600">
                          <span>{doc.type}</span>
                          <span>{doc.size}</span>
                          <span>
                            Uploaded:{" "}
                            {new Date(doc.uploadDate).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <Badge variant="outline">{doc.category}</Badge>
                      <Button size="sm" variant="outline">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Interactions Tab */}
        <TabsContent value="interactions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Interactions ({interactions.length})</span>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  Log Interaction
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {interactions.map((interaction) => (
                  <div key={interaction.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">{interaction.type}</Badge>
                        <h3 className="font-medium">{interaction.subject}</h3>
                      </div>
                      <span className="text-sm text-gray-500">
                        {new Date(interaction.date).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">
                      {interaction.description}
                    </p>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">
                        Contact: {interaction.contact}
                      </span>
                      <span className="text-gray-500">
                        Outcome: {interaction.outcome}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
