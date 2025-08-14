import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Badge } from "./ui/badge";
import {
  Trash2,
  UserPlus,
  Mail,
  Shield,
  User,
  Phone,
  Building,
} from "lucide-react";
import { useAuth } from "./RealAuthProvider";

interface NewUserForm {
  name: string;
  email: string;
  password: string;
  contactNumber: string;
  department: string;
  designation: string;
  role: "admin" | "user";
}

interface DatabaseUser {
  id: string;
  email: string;
  displayName: string;
  role: "admin" | "user";
  emailVerified: boolean;
  createdAt: string;
  contactNumber?: string;
  department?: string;
  designation?: string;
}

export function AdminUserManagement() {
  const { user } = useAuth();
  const [users, setUsers] = useState<DatabaseUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState<NewUserForm>({
    name: "",
    email: "",
    password: "",
    contactNumber: "",
    department: "",
    designation: "",
    role: "user",
  });
  const [createdUser, setCreatedUser] = useState<{
    email: string;
    displayName: string;
  } | null>(null);

  // Fetch users from database
  const fetchUsers = async () => {
    try {
      const response = await fetch("/api/admin/users", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUsers(data.users || []);
      }
    } catch (error) {
      console.error("Failed to fetch users:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === "admin") {
      fetchUsers();
    }
  }, [user]);

  // Handle form submission
  const handleAddUser = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate email and password
    if (!formData.email || !formData.password) {
      alert("Please provide both email and password");
      return;
    }

    if (formData.password.length < 6) {
      alert("Password must be at least 6 characters long");
      return;
    }

    try {
      const response = await fetch("/api/admin/create-user", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
        body: JSON.stringify({
          email: formData.email,
          displayName: formData.name,
          password: formData.password,
          role: formData.role,
          contactNumber: formData.contactNumber,
          department: formData.department,
          designation: formData.designation,
        }),
      });

      if (response.ok) {
        setCreatedUser({
          email: formData.email,
          displayName: formData.name,
        });
        setFormData({
          name: "",
          email: "",
          password: "",
          contactNumber: "",
          department: "",
          designation: "",
          role: "user",
        });
        setShowAddForm(false);
        fetchUsers(); // Refresh the user list
      } else {
        const error = await response.json();
        alert(`Failed to create user: ${error.error}`);
      }
    } catch (error) {
      console.error("Error creating user:", error);
      alert("Failed to create user. Please try again.");
    }
  };

  // Delete user
  const handleDeleteUser = async (userId: string) => {
    if (!confirm("Are you sure you want to delete this user?")) return;

    console.log("🗑️ Deleting user with ID:", userId);

    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });

      const result = await response.json();
      console.log("Delete response:", result);

      if (response.ok) {
        console.log("✅ User deleted successfully");
        fetchUsers(); // Refresh the user list
      } else {
        console.error("❌ Delete failed:", result.error);
        alert(`Failed to delete user: ${result.error}`);
      }
    } catch (error) {
      console.error("Error deleting user:", error);
      alert("Failed to delete user. Please try again.");
    }
  };

  if (user?.role !== "admin") {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-gray-500">
            Access denied. Admin privileges required.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">User Management</h2>
          <p className="text-gray-600">
            Manage company users and add new team members
          </p>
        </div>
        <Button
          onClick={() => setShowAddForm(true)}
          className="bg-blue-600 hover:bg-blue-700"
        >
          <UserPlus className="w-4 h-4 mr-2" />
          Add New Employee
        </Button>
      </div>

      {/* User Created Display */}
      {createdUser && (
        <Card className="border-green-200 bg-green-50">
          <CardHeader>
            <CardTitle className="text-green-800 flex items-center">
              <UserPlus className="w-5 h-5 mr-2" />
              User Created Successfully!
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p>
                <strong>User:</strong> {createdUser.displayName}
              </p>
              <p>
                <strong>Email:</strong>{" "}
                <code className="bg-white px-2 py-1 rounded">
                  {createdUser.email}
                </code>
              </p>
              <p className="text-sm text-green-700">
                The user account has been created successfully with the provided
                credentials.
              </p>
            </div>
            <Button
              onClick={() => setCreatedUser(null)}
              variant="outline"
              size="sm"
              className="mt-3"
            >
              Dismiss
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Add User Form */}
      {showAddForm && (
        <Card>
          <CardHeader>
            <CardTitle>Add New Employee</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleAddUser} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="name">Full Name *</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) =>
                      setFormData({ ...formData, name: e.target.value })
                    }
                    required
                    placeholder="Enter employee's full name"
                  />
                </div>

                <div>
                  <Label htmlFor="email">Email Address *</Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) =>
                      setFormData({ ...formData, email: e.target.value })
                    }
                    required
                    placeholder="Enter email address"
                  />
                </div>

                <div>
                  <Label htmlFor="password">Password *</Label>
                  <Input
                    id="password"
                    type="password"
                    value={formData.password}
                    onChange={(e) =>
                      setFormData({ ...formData, password: e.target.value })
                    }
                    required
                    placeholder="Enter password (min 6 characters)"
                    minLength={6}
                  />
                </div>

                <div>
                  <Label htmlFor="contact">Contact Number *</Label>
                  <Input
                    id="contact"
                    value={formData.contactNumber}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        contactNumber: e.target.value,
                      })
                    }
                    required
                    placeholder="Enter phone number"
                  />
                </div>

                <div>
                  <Label htmlFor="department">Department *</Label>
                  <Input
                    id="department"
                    value={formData.department}
                    onChange={(e) =>
                      setFormData({ ...formData, department: e.target.value })
                    }
                    required
                    placeholder="e.g., Sales, Marketing, IT"
                  />
                </div>

                <div>
                  <Label htmlFor="designation">Designation *</Label>
                  <Input
                    id="designation"
                    value={formData.designation}
                    onChange={(e) =>
                      setFormData({ ...formData, designation: e.target.value })
                    }
                    required
                    placeholder="e.g., Manager, Executive, Developer"
                  />
                </div>

                <div>
                  <Label htmlFor="role">Role *</Label>
                  <Select
                    value={formData.role}
                    onValueChange={(value: "admin" | "user") =>
                      setFormData({ ...formData, role: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="user">User</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex gap-2 pt-4">
                <Button
                  type="submit"
                  className="bg-green-600 hover:bg-green-700"
                >
                  Create User Account
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowAddForm(false)}
                >
                  Cancel
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Users List */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Company Users ({users.length})</CardTitle>
            <div className="flex gap-2">
              <Badge variant="outline">
                {users.filter((u) => u.role === "admin").length} Admins
              </Badge>
              <Badge variant="outline">
                {users.filter((u) => u.role === "user").length} Users
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse p-4 border rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="rounded-full bg-gray-200 h-10 w-10"></div>
                    <div className="space-y-2 flex-1">
                      <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                      <div className="h-3 bg-gray-200 rounded w-1/3"></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : users.length === 0 ? (
            <div className="text-center py-8">
              <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 mb-2">No users found</p>
              <p className="text-sm text-gray-400">
                Add your first team member to get started
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {users.map((dbUser) => (
                <div
                  key={dbUser.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div
                      className={`p-2 rounded-full ${dbUser.role === "admin" ? "bg-purple-100" : "bg-blue-100"}`}
                    >
                      {dbUser.role === "admin" ? (
                        <Shield className="w-4 h-4 text-purple-600" />
                      ) : (
                        <User className="w-4 h-4 text-blue-600" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium">{dbUser.displayName}</h3>
                        <Badge
                          variant={
                            dbUser.role === "admin" ? "default" : "secondary"
                          }
                        >
                          {dbUser.role}
                        </Badge>
                        {dbUser.emailVerified && (
                          <Badge
                            variant="outline"
                            className="text-green-600 border-green-600"
                          >
                            Verified
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mb-1">
                        {dbUser.email}
                      </p>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        {dbUser.department && (
                          <span className="flex items-center gap-1">
                            <Building className="w-3 h-3" />
                            {dbUser.department} • {dbUser.designation}
                          </span>
                        )}
                        {dbUser.contactNumber && (
                          <span className="flex items-center gap-1">
                            <Phone className="w-3 h-3" />
                            {dbUser.contactNumber}
                          </span>
                        )}
                        <span>
                          Created{" "}
                          {new Date(dbUser.createdAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {dbUser.email !== "admin@yitro.com" && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeleteUser(dbUser.id)}
                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
