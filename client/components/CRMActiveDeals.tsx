import React, { useState, useEffect } from "react";
import { useAuth } from "./RealAuthProvider";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { useCRM, type ActiveDeal } from "../contexts/CRMContext";
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
  TrendingUp,
  Filter,
  Plus,
  MoreHorizontal,
  Calendar,
  Edit,
  Trash2,
  Phone,
  Mail,
  DollarSign,
  Building2,
  User,
  AlertCircle,
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

interface ActiveDeal {
  id: string;
  dealName: string;
  businessLine: string;
  associatedAccount: string;
  associatedContact: string;
  closingDate: string;
  probability: number;
  dealValue: number;
  approvedBy: string;
  description: string;
  nextStep: string;
  geo: string;
  entity: string;
  stage: string;
  owner: string;
  ownerId: string;
  createdAt: string;
}

export function CRMActiveDeals() {
  const { user } = useAuth();
  const { deals, addDeal, updateDeal, deleteDeal } = useCRM();
  const [filterStage, setFilterStage] = useState("all");
  const [filterOwner, setFilterOwner] = useState("all");
  const [loading, setLoading] = useState(false);
  const [showNewDealDialog, setShowNewDealDialog] = useState(false);
  const [editingDeal, setEditingDeal] = useState<ActiveDeal | null>(null);
  const [formData, setFormData] = useState<Partial<ActiveDeal>>({});

  useEffect(() => {
    const handleTriggerNewItem = (event: CustomEvent) => {
      if (event.detail.type === "active-deals") {
        resetForm();
        setShowNewDealDialog(true);
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

  const handleSaveDeal = () => {
    console.log("Saving deal:", formData);

    if (!formData.dealName || !formData.businessLine) {
      alert("Deal Name and Business Line are required!");
      return;
    }

    try {
      if (editingDeal) {
        // Update existing deal
        console.log("Updating existing deal:", editingDeal.id);
        updateDeal(editingDeal.id, formData);
        alert("Deal updated successfully!");
      } else {
        // Create new deal
        const newDealData = {
          dealName: formData.dealName || "",
          businessLine: formData.businessLine || "",
          associatedAccount: formData.associatedAccount || "",
          associatedContact: formData.associatedContact || "",
          closingDate:
            formData.closingDate || new Date().toISOString().split("T")[0],
          probability: formData.probability || 50,
          dealValue: formData.dealValue || 0,
          approvedBy: formData.approvedBy || "",
          description: formData.description || "",
          nextStep: formData.nextStep || "",
          geo: formData.geo || "Americas",
          entity: formData.entity || "Yitro Global",
          stage: formData.stage || "Opportunity Identified",
          owner: user?.displayName || "Current User",
          ownerId: user?.id || "current-user",
          createdAt: new Date().toISOString(),
        };
        console.log("Creating new deal:", newDealData);
        addDeal(newDealData);
        alert("Deal created successfully!");
      }

      setShowNewDealDialog(false);
      setEditingDeal(null);
      setFormData({});
    } catch (error) {
      console.error("Error saving deal:", error);
      alert("Error saving deal. Please try again.");
    }
  };

  const handleEditDeal = (deal: ActiveDeal) => {
    console.log("handleEditDeal called with:", deal);
    setEditingDeal(deal);
    setFormData(deal);
    setShowNewDealDialog(true);
  };

  const handleDeleteDeal = (dealId: string) => {
    console.log("handleDeleteDeal called with ID:", dealId);
    if (window.confirm("Are you sure you want to delete this deal?")) {
      deleteDeal(dealId);
      console.log("Deal deleted:", dealId);
      alert("Deal deleted successfully!");
    }
  };

  const resetForm = () => {
    setFormData({});
    setEditingDeal(null);
    setShowNewDealDialog(false);
  };

  const getStageColor = (stage: string) => {
    switch (stage.toLowerCase()) {
      case "opportunity identified":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400";
      case "proposal submitted":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400";
      case "negotiating":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400";
      case "closing":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400";
      case "order won":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
      case "order lost":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400";
    }
  };

  const getProbabilityColor = (probability: number) => {
    if (probability >= 80) return "text-green-600 dark:text-green-400";
    if (probability >= 60) return "text-yellow-600 dark:text-yellow-400";
    if (probability >= 40) return "text-orange-600 dark:text-orange-400";
    return "text-red-600 dark:text-red-400";
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const filteredDeals = deals.filter((deal) => {
    const matchesStage =
      filterStage === "all" || deal.stage.toLowerCase() === filterStage;
    const matchesOwner = filterOwner === "all" || deal.ownerId === filterOwner;

    return matchesStage && matchesOwner;
  });

  // Get unique owners for filter dropdown
  const uniqueOwners = Array.from(
    new Set(deals.map((deal) => ({ id: deal.ownerId, name: deal.owner }))),
  ).reduce(
    (acc, owner) => {
      if (!acc.find((o) => o.id === owner.id)) {
        acc.push(owner);
      }
      return acc;
    },
    [] as { id: string; name: string }[],
  );

  const stages = [
    "Opportunity Identified",
    "Proposal Submitted",
    "Negotiating",
    "Closing",
    "Order Won",
    "Order Lost",
  ];

  const businessLines = [
    "Human Capital",
    "Managed Services",
    "Automation",
    "Support",
    "Product",
    "RCM",
  ];
  const geoOptions = ["Americas", "EMEA", "India", "Philippines", "ANZ"];
  const entityOptions = [
    "Yitro Global",
    "Yitro Tech",
    "Yitro Support",
    "Yitro Health",
  ];

  const totalPipelineValue = filteredDeals.reduce(
    (sum, deal) => sum + deal.dealValue,
    0,
  );
  const avgProbability =
    filteredDeals.length > 0
      ? filteredDeals.reduce((sum, deal) => sum + deal.probability, 0) /
        filteredDeals.length
      : 0;

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <TrendingUp className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              Active Deals
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {user?.role === "admin"
                ? "Manage all active deals across the organization"
                : "Manage your active deals and opportunities"}
            </p>
          </div>
        </div>
        <Dialog open={showNewDealDialog} onOpenChange={setShowNewDealDialog}>
          <DialogTrigger asChild>
            <Button
              className="bg-blue-600 hover:bg-blue-700"
              onClick={() => {
                console.log("New Active Deal button clicked!");
                resetForm();
              }}
            >
              <Plus className="h-4 w-4 mr-2" />
              New Active Deal
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>
                {editingDeal ? "Edit Active Deal" : "Add New Active Deal"}
              </DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="dealName">Deal Name *</Label>
                <Input
                  id="dealName"
                  value={formData.dealName || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, dealName: e.target.value })
                  }
                  placeholder="Enter deal name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="businessLine">Business Line *</Label>
                <Select
                  value={formData.businessLine || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, businessLine: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select business line" />
                  </SelectTrigger>
                  <SelectContent>
                    {businessLines.map((line) => (
                      <SelectItem key={line} value={line}>
                        {line}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="associatedAccount">Associated Account</Label>
                <Input
                  id="associatedAccount"
                  value={formData.associatedAccount || ""}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      associatedAccount: e.target.value,
                    })
                  }
                  placeholder="Account name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="associatedContact">Associated Contact</Label>
                <Input
                  id="associatedContact"
                  value={formData.associatedContact || ""}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      associatedContact: e.target.value,
                    })
                  }
                  placeholder="Contact name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="closingDate">Closing Date</Label>
                <Input
                  id="closingDate"
                  type="date"
                  value={formData.closingDate || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, closingDate: e.target.value })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="probability">Probability (%)</Label>
                <Input
                  id="probability"
                  type="number"
                  min="0"
                  max="100"
                  value={formData.probability || ""}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      probability: parseInt(e.target.value) || 0,
                    })
                  }
                  placeholder="50"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dealValue">Deal Value ($)</Label>
                <Input
                  id="dealValue"
                  type="number"
                  min="0"
                  value={formData.dealValue || ""}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      dealValue: parseInt(e.target.value) || 0,
                    })
                  }
                  placeholder="100000"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="stage">Stage</Label>
                <Select
                  value={formData.stage || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, stage: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select stage" />
                  </SelectTrigger>
                  <SelectContent>
                    {stages.map((stage) => (
                      <SelectItem key={stage} value={stage}>
                        {stage}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="geo">Geo</Label>
                <Select
                  value={formData.geo || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, geo: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select geo" />
                  </SelectTrigger>
                  <SelectContent>
                    {geoOptions.map((geo) => (
                      <SelectItem key={geo} value={geo}>
                        {geo}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="entity">Entity</Label>
                <Select
                  value={formData.entity || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, entity: value })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select entity" />
                  </SelectTrigger>
                  <SelectContent>
                    {entityOptions.map((entity) => (
                      <SelectItem key={entity} value={entity}>
                        {entity}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2 col-span-2">
                <Label htmlFor="description">Description</Label>
                <Input
                  id="description"
                  value={formData.description || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  placeholder="Deal description"
                />
              </div>
              <div className="space-y-2 col-span-2">
                <Label htmlFor="nextStep">Next Step</Label>
                <Input
                  id="nextStep"
                  value={formData.nextStep || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, nextStep: e.target.value })
                  }
                  placeholder="Next action required"
                />
              </div>
            </div>
            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={resetForm}>
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button
                onClick={handleSaveDeal}
                disabled={!formData.dealName || !formData.businessLine}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Save className="h-4 w-4 mr-2" />
                {editingDeal ? "Update" : "Save"} Deal
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
                  Total Active Deals
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {filteredDeals.length}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Pipeline Value
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {formatCurrency(totalPipelineValue)}
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Avg Probability
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {avgProbability.toFixed(1)}%
                </p>
              </div>
              <AlertCircle className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  High Probability
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {filteredDeals.filter((d) => d.probability >= 80).length}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-purple-600" />
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
                  Stage: {filterStage === "all" ? "All" : filterStage}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={() => setFilterStage("all")}>
                  All Stages
                </DropdownMenuItem>
                {stages.map((stage) => (
                  <DropdownMenuItem
                    key={stage}
                    onClick={() => setFilterStage(stage.toLowerCase())}
                  >
                    {stage}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {user?.role === "admin" && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline">
                    <User className="h-4 w-4 mr-2" />
                    Owner:{" "}
                    {filterOwner === "all"
                      ? "All"
                      : uniqueOwners.find((o) => o.id === filterOwner)?.name}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onClick={() => setFilterOwner("all")}>
                    All Owners
                  </DropdownMenuItem>
                  {uniqueOwners.map((owner) => (
                    <DropdownMenuItem
                      key={owner.id}
                      onClick={() => setFilterOwner(owner.id)}
                    >
                      {owner.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Active Deals Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Deal Name</TableHead>
                <TableHead>Business Line</TableHead>
                <TableHead>Account</TableHead>
                <TableHead>Value</TableHead>
                <TableHead>Probability</TableHead>
                <TableHead>Stage</TableHead>
                <TableHead>Closing Date</TableHead>
                {user?.role === "admin" && <TableHead>Owner</TableHead>}
                <TableHead className="w-[50px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredDeals.map((deal) => (
                <TableRow key={deal.id}>
                  <TableCell>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {deal.dealName}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {deal.description}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{deal.businessLine}</Badge>
                  </TableCell>
                  <TableCell>
                    <div>
                      <div className="font-medium">
                        {deal.associatedAccount}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {deal.associatedContact}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">
                    {formatCurrency(deal.dealValue)}
                  </TableCell>
                  <TableCell>
                    <span
                      className={`font-medium ${getProbabilityColor(deal.probability)}`}
                    >
                      {deal.probability}%
                    </span>
                  </TableCell>
                  <TableCell>
                    <Badge className={getStageColor(deal.stage)}>
                      {deal.stage}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center text-sm">
                      <Calendar className="h-3 w-3 mr-1" />
                      {new Date(deal.closingDate).toLocaleDateString()}
                    </div>
                  </TableCell>
                  {user?.role === "admin" && (
                    <TableCell>
                      <div className="flex items-center">
                        <User className="h-3 w-3 mr-1" />
                        {deal.owner}
                      </div>
                    </TableCell>
                  )}
                  <TableCell>
                    <div className="flex space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          console.log("Edit button clicked for deal:", deal);
                          handleEditDeal(deal);
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
                            "Delete button clicked for deal:",
                            deal.id,
                          );
                          handleDeleteDeal(deal.id);
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

      {filteredDeals.length === 0 && (
        <Card>
          <CardContent className="p-6 text-center">
            <TrendingUp className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              No active deals found
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Start by creating your first active deal
            </p>
            {
              <Button className="bg-blue-600 hover:bg-blue-700">
                <Plus className="h-4 w-4 mr-2" />
                Create Active Deal
              </Button>
            }
          </CardContent>
        </Card>
      )}
    </div>
  );
}
