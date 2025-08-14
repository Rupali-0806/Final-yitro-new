import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { useCRM, type Lead } from "../contexts/CRMContext";
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
  Target,
  Filter,
  Plus,
  MoreHorizontal,
  Phone,
  Mail,
  Edit,
  Trash2,
  Star,
  TrendingUp,
  Save,
  X,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

export function CRMLeads() {
  const { leads, addLead, updateLead, deleteLead } = useCRM();
  const [filterStatus, setFilterStatus] = useState("all");
  const [showNewLeadDialog, setShowNewLeadDialog] = useState(false);

  // Debug logging for dialog state changes
  useEffect(() => {
    console.log("showNewLeadDialog state changed:", showNewLeadDialog);
  }, [showNewLeadDialog]);
  const [editingLead, setEditingLead] = useState<Lead | null>(null);
  const [formData, setFormData] = useState<Partial<Lead>>({});

  useEffect(() => {
    const handleTriggerNewItem = (event: CustomEvent) => {
      if (event.detail.type === "leads") {
        resetForm();
        setShowNewLeadDialog(true);
      }
    };

    window.addEventListener(
      "triggerNewItem",
      handleTriggerNewItem as EventListener,
    );
    return () => {
      window.removeEventListener(
        "triggerNewItem",
        handleTriggerNewItem as EventListener,
      );
    };
  }, []);

  // Leads data comes from CRM context

  const handleSaveLead = () => {
    console.log("Saving lead:", formData);

    if (!formData.name || !formData.company) {
      alert("Name and Company are required!");
      return;
    }

    try {
      if (editingLead) {
        // Update existing lead
        console.log("Updating existing lead:", editingLead.id);
        updateLead(editingLead.id, formData);
        alert("Lead updated successfully!");
      } else {
        // Create new lead
        const newLeadData = {
          name: formData.name || "",
          company: formData.company || "",
          title: formData.title || "",
          email: formData.email || "",
          phone: formData.phone || "",
          status: formData.status || "New",
          source: formData.source || "Website",
          score: formData.score || 0,
          value: formData.value || "$0",
          lastActivity: "Just now",
        };
        console.log("Creating new lead:", newLeadData);
        addLead(newLeadData);
        alert("Lead created successfully!");
      }

      setShowNewLeadDialog(false);
      setEditingLead(null);
      setFormData({});
    } catch (error) {
      console.error("Error saving lead:", error);
      alert("Error saving lead. Please try again.");
    }
  };

  const handleEditLead = (lead: Lead) => {
    console.log("handleEditLead called with:", lead);
    setEditingLead(lead);
    setFormData(lead);
    setShowNewLeadDialog(true);
  };

  const handleDeleteLead = (leadId: number) => {
    console.log("handleDeleteLead called with ID:", leadId);
    if (window.confirm("Are you sure you want to delete this lead?")) {
      deleteLead(leadId);
      console.log("Lead deleted:", leadId);
      alert("Lead deleted successfully!");
    }
  };

  const resetForm = () => {
    console.log("resetForm called");
    setFormData({});
    setEditingLead(null);
    setShowNewLeadDialog(false);
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "new":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400";
      case "qualified":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
      case "working":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400";
      case "nurturing":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400";
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-green-600 dark:text-green-400";
    if (score >= 75) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const statuses = ["New", "Qualified", "Working", "Nurturing"];
  const sources = ["Website", "Referral", "Cold Call", "LinkedIn", "Event"];

  const filteredLeads = leads.filter((lead) => {
    const matchesFilter =
      filterStatus === "all" || lead.status.toLowerCase() === filterStatus;

    return matchesFilter;
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Target className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              Leads
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Manage and track your sales leads
            </p>
          </div>
        </div>
        <Dialog open={showNewLeadDialog} onOpenChange={setShowNewLeadDialog}>
          <DialogTrigger asChild>
            <Button
              className="bg-blue-600 hover:bg-blue-700"
              onClick={() => {
                console.log("New Lead button clicked!");
                resetForm();
              }}
            >
              <Plus className="h-4 w-4 mr-2" />
              New Lead
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>
                {editingLead ? "Edit Lead" : "Add New Lead"}
              </DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name *</Label>
                <Input
                  id="name"
                  value={formData.name || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  placeholder="Enter full name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="company">Company *</Label>
                <Input
                  id="company"
                  value={formData.company || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, company: e.target.value })
                  }
                  placeholder="Company name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="title">Title</Label>
                <Input
                  id="title"
                  value={formData.title || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, title: e.target.value })
                  }
                  placeholder="Job title"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, email: e.target.value })
                  }
                  placeholder="email@company.com"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  value={formData.phone || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, phone: e.target.value })
                  }
                  placeholder="+1 (555) 123-4567"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="value">Value</Label>
                <Input
                  id="value"
                  value={formData.value || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, value: e.target.value })
                  }
                  placeholder="$50,000"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="score">Score</Label>
                <Input
                  id="score"
                  type="number"
                  min="0"
                  max="100"
                  value={formData.score || ""}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      score: parseInt(e.target.value) || 0,
                    })
                  }
                  placeholder="85"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="status">Status</Label>
                <Select
                  value={formData.status || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, status: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select status" />
                  </SelectTrigger>
                  <SelectContent>
                    {statuses.map((status) => (
                      <SelectItem key={status} value={status}>
                        {status}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="source">Source</Label>
                <Select
                  value={formData.source || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, source: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select source" />
                  </SelectTrigger>
                  <SelectContent>
                    {sources.map((source) => (
                      <SelectItem key={source} value={source}>
                        {source}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={resetForm}>
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button
                onClick={handleSaveLead}
                disabled={!formData.name || !formData.company}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Save className="h-4 w-4 mr-2" />
                {editingLead ? "Update" : "Save"} Lead
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Leads
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {leads.length}
                </p>
              </div>
              <Target className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Qualified
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {leads.filter((l) => l.status === "Qualified").length}
                </p>
              </div>
              <Star className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Working
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {leads.filter((l) => l.status === "Working").length}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Value
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  $280K
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  <Filter className="h-4 w-4 mr-2" />
                  Status: {filterStatus === "all" ? "All" : filterStatus}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={() => setFilterStatus("all")}>
                  All Status
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setFilterStatus("new")}>
                  New
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setFilterStatus("qualified")}>
                  Qualified
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setFilterStatus("working")}>
                  Working
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setFilterStatus("nurturing")}>
                  Nurturing
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardContent>
      </Card>

      {/* Leads Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Lead Name</TableHead>
                <TableHead>Company</TableHead>
                <TableHead>Contact Info</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Score</TableHead>
                <TableHead>Value</TableHead>
                <TableHead>Last Activity</TableHead>
                <TableHead className="w-[50px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredLeads.map((lead) => (
                <TableRow key={lead.id}>
                  <TableCell>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {lead.name}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {lead.title}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">{lead.company}</TableCell>
                  <TableCell>
                    <div className="space-y-1">
                      <div className="flex items-center text-sm">
                        <Mail className="h-3 w-3 mr-1" />
                        {lead.email}
                      </div>
                      <div className="flex items-center text-sm">
                        <Phone className="h-3 w-3 mr-1" />
                        {lead.phone}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge className={getStatusColor(lead.status)}>
                      {lead.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{lead.source}</TableCell>
                  <TableCell>
                    <span
                      className={`font-semibold ${getScoreColor(lead.score)}`}
                    >
                      {lead.score}
                    </span>
                  </TableCell>
                  <TableCell className="font-medium">{lead.value}</TableCell>
                  <TableCell className="text-sm text-gray-500 dark:text-gray-400">
                    {lead.lastActivity}
                  </TableCell>
                  <TableCell>
                    <div className="flex space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          console.log("Edit button clicked for lead:", lead);
                          handleEditLead(lead);
                        }}
                      >
                        <Edit className="h-4 w-4 mr-1" />
                        Edit
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          console.log(
                            "Delete button clicked for lead:",
                            lead.id,
                          );
                          handleDeleteLead(lead.id);
                        }}
                        className="text-red-600"
                      >
                        <Trash2 className="h-4 w-4 mr-1" />
                        Delete
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
