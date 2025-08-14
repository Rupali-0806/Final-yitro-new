import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  AlertCircle,
  CheckCircle,
  Database,
  Mail,
  Server,
  Settings,
  TestTube,
  Loader2,
} from "lucide-react";
import { useAuth } from "./RealAuthProvider";

interface ConfigStatus {
  database: {
    configured: boolean;
    connected: boolean;
    message: string;
  };
  smtp: {
    configured: boolean;
    connected: boolean;
    message: string;
  };
}

export function ConfigurationValidator() {
  const { user } = useAuth();
  const [status, setStatus] = useState<ConfigStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [testEmail, setTestEmail] = useState("");

  const runConfigTest = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/admin/test-config", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setStatus(data.status);
      } else {
        const error = await response.json();
        console.error("Config test failed:", error);
        setStatus({
          database: {
            configured: false,
            connected: false,
            message: "Failed to test configuration",
          },
          smtp: {
            configured: false,
            connected: false,
            message: "Failed to test configuration",
          },
        });
      }
    } catch (error) {
      console.error("Config test error:", error);
      setStatus({
        database: {
          configured: false,
          connected: false,
          message: "Network error during test",
        },
        smtp: {
          configured: false,
          connected: false,
          message: "Network error during test",
        },
      });
    } finally {
      setLoading(false);
    }
  };

  const sendTestEmail = async () => {
    if (!testEmail) {
      alert("Please enter an email address for the test");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/admin/send-test-email", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
        body: JSON.stringify({ email: testEmail }),
      });

      if (response.ok) {
        alert(
          `Test email sent to ${testEmail}! Check your inbox (and spam folder).`,
        );
      } else {
        const error = await response.json();
        alert(`Failed to send test email: ${error.error}`);
      }
    } catch (error) {
      console.error("Test email error:", error);
      alert("Failed to send test email. Check your network connection.");
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (configured: boolean, connected: boolean) => {
    if (!configured) return <AlertCircle className="h-5 w-5 text-yellow-500" />;
    if (connected) return <CheckCircle className="h-5 w-5 text-green-500" />;
    return <AlertCircle className="h-5 w-5 text-red-500" />;
  };

  const getStatusBadge = (configured: boolean, connected: boolean) => {
    if (!configured) return <Badge variant="outline">Not Configured</Badge>;
    if (connected) return <Badge className="bg-green-500">Connected</Badge>;
    return <Badge variant="destructive">Connection Failed</Badge>;
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
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Production Configuration Validator
          </CardTitle>
          <p className="text-sm text-gray-600">
            Test your database and SMTP configuration for production deployment.
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Button
              onClick={runConfigTest}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <TestTube className="h-4 w-4 mr-2" />
              )}
              Test Configuration
            </Button>

            {status && (
              <div className="space-y-4 mt-6">
                <h3 className="font-semibold">Configuration Status:</h3>

                {/* Database Status */}
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Database className="h-5 w-5 text-blue-600" />
                    <div>
                      <p className="font-medium">Database Connection</p>
                      <p className="text-sm text-gray-600">
                        {status.database.message}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(
                      status.database.configured,
                      status.database.connected,
                    )}
                    {getStatusBadge(
                      status.database.configured,
                      status.database.connected,
                    )}
                  </div>
                </div>

                {/* SMTP Status */}
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Mail className="h-5 w-5 text-green-600" />
                    <div>
                      <p className="font-medium">Email Service (SMTP)</p>
                      <p className="text-sm text-gray-600">
                        {status.smtp.message}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(
                      status.smtp.configured,
                      status.smtp.connected,
                    )}
                    {getStatusBadge(
                      status.smtp.configured,
                      status.smtp.connected,
                    )}
                  </div>
                </div>

                {/* Overall Status */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Server className="h-5 w-5" />
                    <span className="font-medium">Overall Status:</span>
                  </div>
                  {status.database.connected && status.smtp.connected ? (
                    <p className="text-green-600 font-medium">
                      🎉 Production Ready! All systems configured and working.
                    </p>
                  ) : status.database.connected || status.smtp.connected ? (
                    <p className="text-yellow-600 font-medium">
                      ⚠️ Partially Configured. Some features may not work as
                      expected.
                    </p>
                  ) : (
                    <p className="text-red-600 font-medium">
                      🛠️ Development Mode. Using fallback configurations.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Test Email Section */}
      {status?.smtp.connected && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Send Test Email
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="testEmail">Test Email Address</Label>
                <Input
                  id="testEmail"
                  type="email"
                  placeholder="Enter email to receive test message"
                  value={testEmail}
                  onChange={(e) => setTestEmail(e.target.value)}
                />
              </div>
              <Button
                onClick={sendTestEmail}
                disabled={loading || !testEmail}
                variant="outline"
                className="w-full"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Mail className="h-4 w-4 mr-2" />
                )}
                Send Test Email
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration Guide */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration Help</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 text-sm">
            <div>
              <h4 className="font-medium mb-2">📋 Quick Setup:</h4>
              <ol className="list-decimal list-inside space-y-1 text-gray-600">
                <li>
                  Get database URL from Neon, Supabase, or your PostgreSQL
                  provider
                </li>
                <li>
                  Get SMTP credentials (Gmail app password, SendGrid API key,
                  etc.)
                </li>
                <li>Set environment variables using your hosting platform</li>
                <li>Run this configuration test to verify everything works</li>
              </ol>
            </div>

            <div>
              <h4 className="font-medium mb-2">📖 Documentation:</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-600">
                <li>
                  <code>PRODUCTION_SETUP.md</code> - Complete setup guide
                </li>
                <li>
                  <code>CONFIG_EXAMPLES.md</code> - Ready-to-use examples
                </li>
                <li>Use DevServerControl tool for development setup</li>
              </ul>
            </div>

            <div>
              <h4 className="font-medium mb-2">🔧 Common Issues:</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-600">
                <li>
                  <strong>Database:</strong> Check connection string format and
                  SSL settings
                </li>
                <li>
                  <strong>Gmail:</strong> Use app-specific password, not account
                  password
                </li>
                <li>
                  <strong>SendGrid:</strong> Verify API key has proper
                  permissions
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
