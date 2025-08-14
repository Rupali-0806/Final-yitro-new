import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

interface User {
  id: string;
  email: string;
  displayName: string;
  role: "admin" | "user";
  emailVerified: boolean;
  primaryEmail: string;
  createdAt: string;
  lastLogin?: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: (
    email: string,
    password: string,
    displayName: string,
  ) => Promise<{
    success: boolean;
    message?: string;
    requiresVerification?: boolean;
  }>;
  signIn: (
    email: string,
    password: string,
  ) => Promise<{
    success: boolean;
    message?: string;
    requiresVerification?: boolean;
  }>;
  signOut: () => void;
  verifyEmail: (
    token: string,
  ) => Promise<{ success: boolean; message?: string }>;
  requestPasswordReset: (
    email: string,
  ) => Promise<{ success: boolean; message?: string }>;
  resetPassword: (
    token: string,
    newPassword: string,
  ) => Promise<{ success: boolean; message?: string }>;
  changePassword: (
    currentPassword: string,
    newPassword: string,
  ) => Promise<{ success: boolean; message?: string }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within a RealAuthProvider");
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function RealAuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // API base URL
  const API_BASE = "/api/auth";

  // Robust response handling function
  const handleResponse = async (response: Response) => {
    try {
      if (!response || typeof response.ok === 'undefined') {
        return { success: false, error: "Invalid response object" };
      }

      if (!response.ok) {
        // For non-ok responses, try to get error message
        let errorMessage = "Request failed";
        try {
          const errorData = await response.text();
          if (errorData) {
            const parsedError = JSON.parse(errorData);
            errorMessage =
              parsedError.error || parsedError.message || errorMessage;
          }
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        return { success: false, error: errorMessage };
      }

      // For successful responses, parse JSON
      const textResponse = await response.text();
      if (!textResponse || !textResponse.trim()) {
        return { success: false, error: "Empty response from server" };
      }

      const data = JSON.parse(textResponse);
      return { success: true, data };
    } catch (error) {
      console.error("Response handling error:", error);

      // Handle specific JSON parsing errors
      if (error instanceof SyntaxError) {
        return { success: false, error: "Server returned invalid JSON" };
      }

      return { success: false, error: "Invalid response from server" };
    }
  };

  useEffect(() => {
    // Add a small delay to allow the app to fully initialize
    // This helps avoid conflicts with browser extensions
    const timer = setTimeout(() => {
      checkExistingAuth();
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  const checkExistingAuth = async () => {
    try {
      // Check if localStorage is available
      if (typeof Storage === "undefined") {
        console.warn("localStorage not available");
        setLoading(false);
        return;
      }

      const token = localStorage.getItem("auth_token");
      if (!token) {
        setLoading(false);
        return;
      }

      // Add timeout and better error handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch(`${API_BASE}/validate-token`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ token }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle response more defensively
      if (!response) {
        throw new Error("No response received");
      }

      const result = await handleResponse(response);

      if (result.success && result.data?.success && result.data?.data?.user) {
        setUser({
          ...result.data.data.user,
          primaryEmail: result.data.data.user.email,
        });
      } else {
        console.warn("Token validation failed:", result.error);
        try {
          localStorage.removeItem("auth_token");
        } catch (e) {
          console.warn("Could not remove auth token from localStorage");
        }
        setUser(null);
      }
    } catch (error) {
      console.error("Auth check error:", error);

      // Handle different types of errors
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.warn("Auth validation timed out - continuing without authentication");
        } else if (error.message.includes('Failed to fetch')) {
          console.warn("Network error during auth validation - server may be down");
        } else {
          console.warn("Unexpected error during auth validation:", error.message);
        }
      }

      // Clear any invalid tokens and reset user state
      try {
        localStorage.removeItem("auth_token");
      } catch (e) {
        console.warn("Could not remove auth token from localStorage");
      }
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const signUp = async (
    email: string,
    password: string,
    displayName: string,
  ) => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE}/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password, displayName }),
      });

      const result = await handleResponse(response);

      if (result.success && result.data.success) {
        localStorage.setItem("auth_token", result.data.data.token);
        setUser({
          ...result.data.data.user,
          primaryEmail: result.data.data.user.email,
        });

        return {
          success: true,
          message: result.data.data.message,
          requiresVerification: !result.data.data.user.emailVerified,
        };
      } else {
        return {
          success: false,
          message: result.error || "Failed to create account",
        };
      }
    } catch (error) {
      console.error("SignUp error:", error);
      return {
        success: false,
        message: "Network error. Please try again.",
      };
    } finally {
      setLoading(false);
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      setLoading(true);

      // Add timeout for signin request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch(`${API_BASE}/signin`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const result = await handleResponse(response);

      if (result.success && result.data.success) {
        localStorage.setItem("auth_token", result.data.data.token);
        setUser({
          ...result.data.data.user,
          primaryEmail: result.data.data.user.email,
        });

        return {
          success: true,
          message: result.data.data.message,
        };
      } else {
        const errorMessage = result.error || "Failed to sign in";
        return {
          success: false,
          message: errorMessage,
          requiresVerification: errorMessage.includes("verify"),
        };
      }
    } catch (error) {
      console.error("SignIn error:", error);

      let errorMessage = "Network error. Please try again.";
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = "Request timed out. Please try again.";
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = "Unable to connect to server. Please check your connection.";
        }
      }

      return {
        success: false,
        message: errorMessage,
      };
    } finally {
      setLoading(false);
    }
  };

  const signOut = () => {
    localStorage.removeItem("auth_token");
    setUser(null);
  };

  const verifyEmail = async (token: string) => {
    try {
      const response = await fetch(`${API_BASE}/verify-email`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ token }),
      });

      const result = await handleResponse(response);

      if (result.success && result.data.success) {
        if (user) {
          setUser({ ...user, emailVerified: true });
        }
        return {
          success: true,
          message: result.data.data.message,
        };
      } else {
        return {
          success: false,
          message: result.error || "Failed to verify email",
        };
      }
    } catch (error) {
      console.error("Email verification error:", error);
      return {
        success: false,
        message: "Network error. Please try again.",
      };
    }
  };

  const requestPasswordReset = async (email: string) => {
    try {
      const response = await fetch(`${API_BASE}/request-password-reset`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      });

      const result = await handleResponse(response);

      return {
        success: result.success && result.data.success,
        message:
          result.success && result.data.success
            ? result.data.data.message
            : result.error || "Failed to send reset email",
      };
    } catch (error) {
      console.error("Password reset request error:", error);
      return {
        success: false,
        message: "Network error. Please try again.",
      };
    }
  };

  const resetPassword = async (token: string, newPassword: string) => {
    try {
      const response = await fetch(`${API_BASE}/reset-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ token, newPassword }),
      });

      const result = await handleResponse(response);

      return {
        success: result.success && result.data.success,
        message:
          result.success && result.data.success
            ? result.data.data.message
            : result.error || "Failed to reset password",
      };
    } catch (error) {
      console.error("Password reset error:", error);
      return {
        success: false,
        message: "Network error. Please try again.",
      };
    }
  };

  const changePassword = async (
    currentPassword: string,
    newPassword: string,
  ) => {
    try {
      const token = localStorage.getItem("auth_token");
      if (!token) {
        return {
          success: false,
          message: "Not authenticated",
        };
      }

      const response = await fetch(`${API_BASE}/change-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ currentPassword, newPassword }),
      });

      const result = await handleResponse(response);

      return {
        success: result.success && result.data.success,
        message:
          result.success && result.data.success
            ? result.data.data.message
            : result.error || "Failed to change password",
      };
    } catch (error) {
      console.error("Change password error:", error);
      return {
        success: false,
        message: "Network error. Please try again.",
      };
    }
  };

  const value: AuthContextType = {
    user,
    loading,
    signUp,
    signIn,
    signOut,
    verifyEmail,
    requestPasswordReset,
    resetPassword,
    changePassword,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Compatibility hook for existing components
export function useUser() {
  const { user } = useAuth();
  return user;
}
