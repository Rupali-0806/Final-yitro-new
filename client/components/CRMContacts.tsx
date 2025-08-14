import React, { useState, useEffect } from "react";
import { useAuth } from "./RealAuthProvider";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { useCRM, type Contact } from "../contexts/CRMContext";
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
import {
  Users,
  Filter,
  Plus,
  MoreHorizontal,
  Phone,
  Mail,
  Edit,
  Trash2,
  MapPin,
  Building2,
  User,
  Save,
  X,
} from "lucide-react";

export function CRMContacts() {
  console.log("CRMContacts component loaded");
  const { user } = useAuth();
  const { contacts, addContact, updateContact, deleteContact } = useCRM();
  const [filterStatus, setFilterStatus] = useState("all");
  const [filterOwner, setFilterOwner] = useState("all");
  const [loading, setLoading] = useState(false);
  const [showNewContactDialog, setShowNewContactDialog] = useState(false);
  const [editingContact, setEditingContact] = useState<Contact | null>(null);
  const [formData, setFormData] = useState<Partial<Contact>>({});

  useEffect(() => {
    const handleTriggerNewItem = (event: CustomEvent) => {
      if (event.detail.type === "contacts") {
        resetForm();
        setShowNewContactDialog(true);
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

  const handleSaveContact = () => {
    console.log("Saving contact:", formData);

    if (!formData.firstName || !formData.lastName) {
      alert("First Name and Last Name are required!");
      return;
    }

    try {
      if (editingContact) {
        // Update existing contact
        console.log("Updating existing contact:", editingContact.id);
        updateContact(editingContact.id, formData);
        alert("Contact updated successfully!");
      } else {
        // Create new contact
        const newContactData = {
          firstName: formData.firstName || "",
          lastName: formData.lastName || "",
          title: formData.title,
          associatedAccount: formData.associatedAccount,
          emailAddress: formData.emailAddress,
          deskPhone: formData.deskPhone,
          mobilePhone: formData.mobilePhone,
          city: formData.city,
          state: formData.state,
          country: formData.country,
          timeZone: formData.timeZone,
          source: formData.source,
          owner: user?.displayName || "Current User",
          ownerId: user?.id || "current-user",
          status: formData.status,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        console.log("Creating new contact:", newContactData);
        addContact(newContactData);
        alert("Contact created successfully!");
      }

      setShowNewContactDialog(false);
      setEditingContact(null);
      setFormData({});
    } catch (error) {
      console.error("Error saving contact:", error);
      alert("Error saving contact. Please try again.");
    }
  };

  const handleEditContact = (contact: Contact) => {
    console.log("Editing contact:", contact);
    setEditingContact(contact);
    setFormData(contact);
    setShowNewContactDialog(true);
  };

  const handleDeleteContact = (contactId: string) => {
    console.log("Attempting to delete contact:", contactId);
    if (window.confirm("Are you sure you want to delete this contact?")) {
      deleteContact(contactId);
      console.log("Contact deleted:", contactId);
      alert("Contact deleted successfully!");
    }
  };

  const resetForm = () => {
    setFormData({});
    setEditingContact(null);
    setShowNewContactDialog(false);
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case "active deal":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
      case "prospect":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400";
      case "suspect":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400";
      case "do not call":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400";
    }
  };

  const filteredContacts = contacts.filter((contact) => {
    const matchesStatus =
      filterStatus === "all" || contact.status?.toLowerCase() === filterStatus;
    const matchesOwner =
      filterOwner === "all" || contact.ownerId === filterOwner;

    return matchesStatus && matchesOwner;
  });

  // Get unique owners for filter dropdown
  const uniqueOwners = Array.from(
    new Set(
      contacts.map((contact) => ({ id: contact.ownerId, name: contact.owner })),
    ),
  ).reduce(
    (acc, owner) => {
      if (!acc.find((o) => o.id === owner.id)) {
        acc.push(owner);
      }
      return acc;
    },
    [] as { id: string; name: string }[],
  );

  const statuses = ["Suspect", "Prospect", "Active Deal", "Do Not Call"];
  const sources = ["Data Research", "Referral", "Event"];

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
          <Users className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              Contacts
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {user?.role === "admin"
                ? "Manage all contacts across the organization"
                : "Manage your assigned contacts and relationships"}
            </p>
          </div>
        </div>
        <Dialog
          open={showNewContactDialog}
          onOpenChange={setShowNewContactDialog}
        >
          <DialogTrigger asChild>
            <Button
              className="bg-blue-600 hover:bg-blue-700"
              onClick={() => resetForm()}
            >
              <Plus className="h-4 w-4 mr-2" />
              New Contact
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>
                {editingContact ? "Edit Contact" : "Add New Contact"}
              </DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="firstName">First Name *</Label>
                <Input
                  id="firstName"
                  value={formData.firstName || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, firstName: e.target.value })
                  }
                  placeholder="Enter first name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName">Last Name *</Label>
                <Input
                  id="lastName"
                  value={formData.lastName || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, lastName: e.target.value })
                  }
                  placeholder="Enter last name"
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
                  placeholder="Company name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="emailAddress">Email Address</Label>
                <Input
                  id="emailAddress"
                  type="email"
                  value={formData.emailAddress || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, emailAddress: e.target.value })
                  }
                  placeholder="email@company.com"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="deskPhone">Desk Phone</Label>
                <Input
                  id="deskPhone"
                  value={formData.deskPhone || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, deskPhone: e.target.value })
                  }
                  placeholder="+1 (555) 123-4567"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="mobilePhone">Mobile Phone</Label>
                <Input
                  id="mobilePhone"
                  value={formData.mobilePhone || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, mobilePhone: e.target.value })
                  }
                  placeholder="+1 (555) 123-4567"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="city">City</Label>
                <Input
                  id="city"
                  value={formData.city || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, city: e.target.value })
                  }
                  placeholder="City"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="state">State</Label>
                <Input
                  id="state"
                  value={formData.state || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, state: e.target.value })
                  }
                  placeholder="State"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="country">Country</Label>
                <Input
                  id="country"
                  value={formData.country || ""}
                  onChange={(e) =>
                    setFormData({ ...formData, country: e.target.value })
                  }
                  placeholder="Country"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="source">Source</Label>
                <Select
                  value={formData.source || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, source: value as any })
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
              <div className="space-y-2">
                <Label htmlFor="status">Status</Label>
                <Select
                  value={formData.status || ""}
                  onValueChange={(value) =>
                    setFormData({ ...formData, status: value as any })
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
            </div>
            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={resetForm}>
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button
                onClick={handleSaveContact}
                disabled={!formData.firstName || !formData.lastName}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <Save className="h-4 w-4 mr-2" />
                {editingContact ? "Update" : "Save"} Contact
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
                  Total Contacts
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {contacts.length}
                </p>
              </div>
              <Users className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Active Deals
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {contacts.filter((c) => c.status === "Active Deal").length}
                </p>
              </div>
              <Building2 className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Prospects
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {contacts.filter((c) => c.status === "Prospect").length}
                </p>
              </div>
              <User className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  This Month
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {
                    contacts.filter((c) => {
                      const created = new Date(c.createdAt);
                      const now = new Date();
                      return (
                        created.getMonth() === now.getMonth() &&
                        created.getFullYear() === now.getFullYear()
                      );
                    }).length
                  }
                </p>
              </div>
              <Plus className="h-8 w-8 text-purple-600" />
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
                  All Statuses
                </DropdownMenuItem>
                {statuses.map((status) => (
                  <DropdownMenuItem
                    key={status}
                    onClick={() => setFilterStatus(status.toLowerCase())}
                  >
                    {status}
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

      {/* Contacts Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Title</TableHead>
                <TableHead>Account</TableHead>
                <TableHead>Contact Info</TableHead>
                <TableHead>Location</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Source</TableHead>
                {user?.role === "admin" && <TableHead>Owner</TableHead>}
                <TableHead className="w-[50px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredContacts.map((contact) => (
                <TableRow key={contact.id}>
                  <TableCell>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {contact.firstName} {contact.lastName}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        ID: {contact.id}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>{contact.title || "N/A"}</TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      <Building2 className="h-3 w-3 mr-1" />
                      {contact.associatedAccount || "N/A"}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="space-y-1">
                      {contact.emailAddress && (
                        <div className="flex items-center text-sm">
                          <Mail className="h-3 w-3 mr-1" />
                          {contact.emailAddress}
                        </div>
                      )}
                      {contact.mobilePhone && (
                        <div className="flex items-center text-sm">
                          <Phone className="h-3 w-3 mr-1" />
                          {contact.mobilePhone}
                        </div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center text-sm">
                      <MapPin className="h-3 w-3 mr-1" />
                      {contact.city && contact.state
                        ? `${contact.city}, ${contact.state}`
                        : contact.city || contact.state || "N/A"}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge className={getStatusColor(contact.status || "")}>
                      {contact.status || "N/A"}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{contact.source || "N/A"}</Badge>
                  </TableCell>
                  {user?.role === "admin" && (
                    <TableCell>
                      <div className="flex items-center">
                        <User className="h-3 w-3 mr-1" />
                        {contact.owner}
                      </div>
                    </TableCell>
                  )}
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={() => handleEditContact(contact)}
                        >
                          <Edit className="h-4 w-4 mr-2" />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => handleDeleteContact(contact.id)}
                          className="text-red-600"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {filteredContacts.length === 0 && (
        <Card>
          <CardContent className="p-6 text-center">
            <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              No contacts found
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Start by adding your first contact
            </p>
            {
              <Button
                className="bg-blue-600 hover:bg-blue-700"
                onClick={() => setShowNewContactDialog(true)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Contact
              </Button>
            }
          </CardContent>
        </Card>
      )}
    </div>
  );
}
